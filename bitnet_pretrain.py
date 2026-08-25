#!/usr/bin/env python3
# bitnet_pretrain.py — native 1.58-bit bidirectional ENCODER for MLM pretraining.
#
# Stage-1 of the two-stage 1.58-bit PII plan (see EVIDENCE_MAP.md). This provides a
# reusable 1.58-bit encoder backbone that:
#   * trains via masked language modeling (MLM) on general text -> general language
#     knowledge (the thing the from-scratch 42M PII model lacked and overfit).
#   * is later reused as the backbone of a label-conditioned span head (Stage 2).
#
# Implements the BitNet b1.58 recipe (arxiv:2504.12285 §2-3):
#   * BitLinear: weights -> ternary {-1,0,+1} via ABSMEAN; activations -> per-token
#     int8 via ABSMAX (on the input side).
#   * subLN normalization; ReLU^2 (not SwiGLU); RoPE; no bias.
#   * dropout to regularize (PII fine-tune data is tiny).
#   * bf16/fp32 MASTER weights kept by optimizer; ternary applied on forward only (STE).
#
# Run (CPU smoke):  python -c "from bitnet_pretrain import *; smoke()"
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ quant prims
class WeightQuant(torch.autograd.Function):
    """Absmean ternary {-1,0,+1} quantization with straight-through estimator."""
    @staticmethod
    def forward(ctx, weight):
        dtype = weight.dtype
        w = weight.float()
        scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
        w = (w * scale).round().clamp(-1, 1) / scale
        return w.to(dtype)
    @staticmethod
    def backward(ctx, g):
        return g.clone()


class ActQuant(torch.autograd.Function):
    """Per-token absmax symmetric int8 quantization with STE."""
    @staticmethod
    def forward(ctx, a):
        dtype = a.dtype
        a = a.float()
        scale = 127.0 / a.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
        a = (a * scale).round().clamp(-128, 127) / scale
        return a.to(dtype)
    @staticmethod
    def backward(ctx, g):
        return g.clone()


class BitRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        d = x.dtype; x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(d)


class BitLinear(nn.Linear):
    """1.58-bit quantized linear (absmean ternary weights + per-token int8 activations)."""
    def __init__(self, in_f, out_f, bias=False, use_sub_norm=False):
        super().__init__(in_f, out_f, bias)
        self.use_sub_norm = use_sub_norm
        self.sub_norm = BitRMSNorm(in_f) if use_sub_norm else None
    def forward(self, x):
        if self.use_sub_norm:
            x = self.sub_norm(x)
        w = WeightQuant.apply(self.weight)
        x = ActQuant.apply(x)
        out = F.linear(x, w)
        if self.bias is not None:
            out = out + self.bias.view(1, -1)
        return out


class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv)
    def forward(self, pos):
        pos = pos.float().reshape(-1)
        freq = torch.outer(pos, self.inv_freq)            # [S, D/2]
        cos = torch.cat([torch.cos(freq), torch.cos(freq)], -1)
        sin = torch.cat([torch.sin(freq), torch.sin(freq)], -1)
        return cos, sin


def rotate_half(x):
    a, b = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-b, a), -1)


def apply_rot(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0); sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos + rotate_half(q) * sin), (k * cos + rotate_half(k) * sin)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.num_heads; self.hd = c.hidden_size // c.num_heads
        self.kv_h = getattr(c, "num_kv_heads", self.h)
        self.rep = self.h // self.kv_h
        self.q = BitLinear(c.hidden_size, self.h * self.hd, bias=False)
        self.k = BitLinear(c.hidden_size, self.kv_h * self.hd, bias=False)
        self.v = BitLinear(c.hidden_size, self.kv_h * self.hd, bias=False)
        self.o = BitLinear(self.h * self.hd, c.hidden_size, bias=False)
        self.rot = Rotary(self.hd)
    def forward(self, x, mask):
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.h, self.hd).transpose(1, 2)
        k = self.k(x).view(B, S, self.kv_h, self.hd).transpose(1, 2)
        v = self.v(x).view(B, S, self.kv_h, self.hd).transpose(1, 2)
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        cos, sin = self.rot(pos)
        q, k = apply_rot(q, k, cos, sin)
        if self.rep > 1:
            k = k.repeat_interleave(self.rep, 1); v = v.repeat_interleave(self.rep, 1)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(a.transpose(1, 2).reshape(B, S, self.h * self.hd))


class MLP(nn.Module):
    """ReLU^2 FFN per the BitNet b1.58 recipe (not SwiGLU) + dropout."""
    def __init__(self, c):
        super().__init__(); i = c.intermediate_size
        self.gate = BitLinear(c.hidden_size, i); self.up = BitLinear(c.hidden_size, i)
        self.down = BitLinear(i, c.hidden_size, use_sub_norm=True)
        self.drop = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
    def forward(self, x):
        h = torch.relu(self.gate(x)).pow(2) * self.up(x)
        return self.drop(self.down(h))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.an = BitRMSNorm(c.hidden_size); self.attn = Attention(c)
        self.mn = BitRMSNorm(c.hidden_size); self.mlp = MLP(c)
        self.drop = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
    def forward(self, x, mask):
        h = x + self.drop(self.attn(self.an(x), mask))
        return h + self.drop(self.mlp(self.mn(h)))


@dataclass
class EncoderConfig:
    vocab_size: int = 16384
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    intermediate_size: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1
    rms_norm_eps: float = 1e-6


class BitnetEncoder(nn.Module):
    """Bidirectional 1.58-bit encoder backbone (no task head)."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.num_layers)])
        self.norm = BitRMSNorm(c.hidden_size)
        self._init()
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.normal_(self.embed.weight, std=0.02)
    def forward(self, input_ids, attention_mask=None):
        x = self.embed(input_ids)
        for blk in self.blocks:
            x = blk(x, None)   # bidirectional (no causal mask)
        return self.norm(x)    # [B, S, H]
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


class MLMHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(c.hidden_size),
            nn.Linear(c.hidden_size, c.hidden_size),
            nn.GELU(),
            nn.Linear(c.hidden_size, c.vocab_size),
        )
    def forward(self, hidden):
        return self.mlp(hidden)


class BitnetMLM(nn.Module):
    """Encoder + MLM head, used for stage-1 pretraining."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.encoder = BitnetEncoder(c)
        self.mlm = MLMHead(c)
        self._init_mlm()
    def _init_mlm(self):
        for m in self.mlm.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, input_ids, labels=None):
        hidden = self.encoder(input_ids)
        logits = self.mlm(hidden)   # [B, S, V]
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1),
                                   ignore_index=-100)
        return logits, loss
    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------ smoke test
def smoke():
    torch.manual_seed(0)
    c = EncoderConfig(vocab_size=2048, hidden_size=128, num_layers=2, num_heads=4,
                      num_kv_heads=4, intermediate_size=512, max_seq_len=64, dropout=0.1)
    m = BitnetMLM(c)
    print("params:", round(m.param_count()/1e6, 3), "M")
    x = torch.randint(0, 2048, (2, 64))
    labels = x.clone()
    labels[:, :5] = -100
    logits, loss = m(x, labels=labels)
    loss.backward()
    print("logits", tuple(logits.shape), "loss", round(loss.item(), 4))
    gn = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print("grad nonzero:", gn > 0)
    w = torch.randn(64, 64) * 0.02
    wq = WeightQuant.apply(w)
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    print("ternary code set:", sorted(set((wq * scale).round().unique().tolist())))
    print("SMOKE OK")


if __name__ == "__main__":
    smoke()
