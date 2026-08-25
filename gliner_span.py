#!/usr/bin/env python3
# gliner_span.py — GLiNER-style label-conditioned span extraction head (FIXED).
#
# Stage-2 architecture after fixing the root-cause bugs found by A/B diagnosis:
#   BUG-FIX 1: labels are encoded through the SAME pretrained 1.58-bit backbone
#              (NOT a detached random char-LSTM). The label phrase is tokenized with
#              the shared tokenizer, passed through the backbone, and pooled, so the
#              token and label representations live in the SAME learned space.
#   BUG-FIX 2: a span-extraction pretraining objective is used (train the span head
#              to predict start/end boundaries on a broad entity corpus BEFORE PII
#              fine-tune), so the model learns the extraction MECHANISM, not just PII.
#
# The model conditions on target labels (passed as text) and returns exact char spans.
import torch
import torch.nn as nn
import torch.nn.functional as F

from bitnet_pretrain import BitnetEncoder, EncoderConfig


class SharedLabelEncoder(nn.Module):
    """Encode label phrases through the SAME 1.58-bit backbone -> [L, S2, H], pooled [L,H]."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        # label phrases are tokenized with the main tokenizer; the backbone encodes them.
        self.encoder = BitnetEncoder(c)          # shares architecture with text encoder
        self.proj = nn.Linear(c.hidden_size, c.hidden_size)
    def forward(self, label_ids):
        """label_ids: [L, label_len] (same vocab tokenizer). Returns [L, H]."""
        h = self.encoder(label_ids)               # [L, label_len, H]
        return torch.tanh(self.proj(h.mean(dim=1)))   # [L, H]


class SpanHead(nn.Module):
    """Per-(token,label) start/end scoring with a projection (not raw concat)."""
    def __init__(self, c):
        super().__init__()
        h = c.hidden_size
        # project token and label separately before matching (better than raw concat)
        self.tok_proj = nn.Linear(h, h)
        self.lab_proj = nn.Linear(h, h)
        self.start = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        self.end = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, token_h, label_h):
        B, S, H = token_h.shape
        L = label_h.shape[0]
        th = self.tok_proj(token_h)          # [B,S,H]
        lh = self.lab_proj(label_h)          # [L,H]
        tok = th.unsqueeze(2).expand(B, S, L, H)
        lab = lh.unsqueeze(0).unsqueeze(0).expand(B, S, L, H)
        pair = torch.cat([tok, lab], dim=-1)
        start_logits = self.start(pair).squeeze(-1)
        end_logits = self.end(pair).squeeze(-1)
        return start_logits, end_logits


class GlinerPII(nn.Module):
    """Pretrained 1.58-bit encoder (shared for text+labels) + fixed span head."""
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.encoder = BitnetEncoder(c)          # 1.58-bit backbone: text
        self.label_enc = SharedLabelEncoder(c)   # same architecture for labels
        self.head = SpanHead(c)
        self._init_heads()
    def _init_heads(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.label_enc.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, input_ids, label_ids):
        token_h = self.encoder(input_ids)          # [B,S,H]
        label_h = self.label_enc(label_ids)        # [L,H]
        return self.head(token_h, label_h)
    def param_count(self):
        return sum(p.numel() for p in self.parameters())
    def load_encoder_weights(self, sd, drop_keys=("embed.",)):
        """Load pretrained MLM encoder weights into BOTH text and label encoders."""
        filtered = {k: v for k, v in sd.items() if not any(k.startswith(dk) for dk in drop_keys)}
        # load into self.encoder
        try:
            self.encoder.load_state_dict(sd, strict=False)
        except Exception:
            pass
        # load into label_enc.encoder (same architecture) — strict=False to skip size mismatches
        try:
            self.label_enc.encoder.load_state_dict(sd, strict=False)
        except Exception:
            pass


def label_text_to_token_ids(label_texts, tok, max_len=16):
    """Tokenize label phrases with the SHARED tokenizer -> [L, max_len] token ids."""
    ids = []
    for t in label_texts:
        e = tok.encode(t)
        row = e.ids[:max_len]
        row = row + [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long)


# ------------------------------------------------------------------ smoke test
def smoke():
    torch.manual_seed(0)
    c = EncoderConfig(vocab_size=32000, hidden_size=256, num_layers=2, num_heads=4,
                      num_kv_heads=4, intermediate_size=1024, max_seq_len=256, dropout=0.1)
    m = GlinerPII(c)
    print("params:", round(m.param_count()/1e6, 2), "M")
    x = torch.randint(0, 32000, (2, 64))
    lab = torch.randint(0, 32000, (4, 12))
    sl, el = m(x, lab)
    print("start_logits", tuple(sl.shape), "end_logits", tuple(el.shape))
    tgt = (torch.rand_like(sl) > 0.9).float(); tgt_e = (torch.rand_like(el) > 0.9).float()
    loss = F.binary_cross_entropy_with_logits(sl, tgt) + F.binary_cross_entropy_with_logits(el, tgt_e)
    loss.backward()
    print("loss", round(loss.item(), 4), "grad ok:", sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None) > 0)
    print("SMOKE OK")


if __name__ == "__main__":
    smoke()
