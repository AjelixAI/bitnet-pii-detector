#!/usr/bin/env python3
# fp_encoder.py — full-precision CONTROL encoder for the 1.58-bit A/B test.
#
# Question: does 1.58-bit quantization hurt PII quality, or is the 0.70-0.77 F1
# ceiling set by data/recipe (not precision)?
#
# This is a clean control: the SAME architecture & pretrained master weights as
# bitnet_pretrain.BitnetEncoder (fp continuous, since BitNet keeps fp master and
# ternary is forward-only), but with plain nn.Linear / nn.LayerNorm — NO quant in
# the forward pass. Fine-tuning it identically isolates the quantization effect.
#
# Run the 1.58-bit BIOES fine-tune path but swap in this model.
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_pretrain import EncoderConfig, Rotary, rotate_half, apply_rot


class FPRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__(); self.weight = nn.Parameter(torch.ones(dim)); self.eps = eps
    def forward(self, x):
        d = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(d)


class FPAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h = c.num_heads; self.hd = c.hidden_size // c.num_heads
        self.kv_h = getattr(c, "num_kv_heads", self.h)
        self.rep = self.h // self.kv_h
        self.q = nn.Linear(c.hidden_size, self.h * self.hd, bias=False)
        self.k = nn.Linear(c.hidden_size, self.kv_h * self.hd, bias=False)
        self.v = nn.Linear(c.hidden_size, self.kv_h * self.hd, bias=False)
        self.o = nn.Linear(self.h * self.hd, c.hidden_size, bias=False)
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


class FPMLP(nn.Module):
    def __init__(self, c):
        super().__init__(); i = c.intermediate_size
        self.gate = nn.Linear(c.hidden_size, i); self.up = nn.Linear(c.hidden_size, i)
        self.down = nn.Linear(i, c.hidden_size)
        self.drop = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
    def forward(self, x):
        h = torch.relu(self.gate(x)).pow(2) * self.up(x)
        return self.drop(self.down(h))


class FPBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.an = FPRMSNorm(c.hidden_size); self.attn = FPAttention(c)
        self.mn = FPRMSNorm(c.hidden_size); self.mlp = FPMLP(c)
        self.drop = nn.Dropout(c.dropout) if c.dropout > 0 else nn.Identity()
    def forward(self, x, mask):
        h = x + self.drop(self.attn(self.an(x), mask))
        return h + self.drop(self.mlp(self.mn(h)))


class FPEncoder(nn.Module):
    """Full-precision (no quant) encoder, same architecture/weights as BitnetEncoder."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList([FPBlock(c) for _ in range(c.num_layers)])
        self.norm = FPRMSNorm(c.hidden_size)
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
            x = blk(x, None)
        return self.norm(x)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    def load_bitnet_weights(self, sd):
        """Load pretrained BitnetEncoder fp master weights (keys match 1:1)."""
        self.load_state_dict(sd, strict=False)


if __name__ == "__main__":
    torch.manual_seed(0)
    c = EncoderConfig(vocab_size=65000, hidden_size=1024, num_layers=16, num_heads=16,
                      num_kv_heads=8, intermediate_size=4096, max_seq_len=512, dropout=0.1)
    m = FPEncoder(c)
    print("FPEncoder params:", round(m.param_count()/1e6, 1), "M")
    x = torch.randint(0, 65000, (2, 64))
    h = m(x)
    print("output:", tuple(h.shape), "no quant; fp forward OK")
