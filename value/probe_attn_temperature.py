"""Measure the ACTUAL pre-softmax attention logits + softmax entropy of the
trained anti-collapse A/B arms, to test the claimed mechanism:

  token norm collapse -> logit ~ norm^2 -> softmax flattens to uniform -> hand
  structure averaged away.

Reads the arm checkpoints written by train_value_collapse_ab.py. Reproduces
nn.MultiheadAttention's internals by hand (in_proj_weight split q/k/v, 4 heads,
head_dim 16) so we can see the logits BEFORE the softmax.

2026-08-05 findings (this probe + probe_qk_gamma.py + trace_mse_vs_collapse.py):
plain/featref/B/D entropy EXACTLY uniform (1.0000, logit spread ~5e-7); C is
SHARPER than init (0.8953 < 0.9554). Main collapse driver is W_q/W_k decay to
~0 (x1.1e5 of the logit loss) after softmax uniformity zeroes q/k grads —
the emb-norm term contributes only x20. gamma(LN) shrinks under blanket weight
decay (C 0.224, BCD 0.063) — watch it in any pre-LN deployment.
"""


import bootstrap  # noqa: F401

import numpy as np
import torch

from value_model import ValueNet, D
from value_model_v2 import ValueNetV2

DEV = "cpu"
CKPT = bootstrap.CKPT_DIR
NHEAD, HDIM = 4, D // 4

FACTORY = {
    "plain":   lambda: ValueNet(extra_dim=0),
    "featref": lambda: ValueNet(extra_dim=1),
    "B":       lambda: ValueNetV2(split_emb=True),
    "C":       lambda: ValueNetV2(pre_ln=True, bag_mean=True),
    "D":       lambda: ValueNetV2(residual=True, own_count=True),
    "BCD":     lambda: ValueNetV2(split_emb=True, pre_ln=True, bag_mean=True,
                                  residual=True, own_count=True),
}


def tokens_of(net, ids):
    """The exact tensor that gets fed to hand_attn as q/k/v."""
    if isinstance(net, ValueNetV2):
        h = net.hand_emb(ids)
        return net.tok_ln(h) if net.tok_ln is not None else h
    return net.card_emb(ids)


@torch.no_grad()
def attn_stats(net, ids, pad):
    t = tokens_of(net, ids)                                  # (B,L,D)
    mha = net.hand_attn
    Wq, Wk = mha.in_proj_weight[:D], mha.in_proj_weight[D:2 * D]
    bq, bk = mha.in_proj_bias[:D], mha.in_proj_bias[D:2 * D]
    B, L, _ = t.shape
    q = (t @ Wq.T + bq).view(B, L, NHEAD, HDIM).transpose(1, 2)   # (B,H,L,hd)
    k = (t @ Wk.T + bk).view(B, L, NHEAD, HDIM).transpose(1, 2)
    logit = (q @ k.transpose(-1, -2)) / (HDIM ** 0.5)             # (B,H,L,L)

    valid = (~pad).float()                                        # (B,L)
    nval = valid.sum(1)                                           # (B,)
    kmask = valid[:, None, None, :]                               # mask over KEYS
    qmask = valid[:, None, :, None]                               # mask over QUERIES

    # spread of logits across valid keys, per (row, head) — the softmax's input range
    big = torch.where(kmask.bool(), logit, torch.tensor(-1e30))
    small = torch.where(kmask.bool(), logit, torch.tensor(1e30))
    spread = big.amax(-1) - small.amin(-1)                        # (B,H,L)

    p = torch.softmax(big, dim=-1)
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)                 # (B,H,L) nats
    # normalise: 1.0 == uniform over the valid keys, 0.0 == one-hot
    ent_n = ent / nval.clamp(min=2).log()[:, None, None]

    w = (qmask.squeeze(-1) * torch.ones_like(spread))             # weight valid queries
    wsum = w.sum()
    tok_rms = (t.pow(2).mean(-1).sqrt() * valid).sum() / valid.sum()
    return {
        "tok_rms": float(tok_rms),
        "tok_norm": float(((t.norm(dim=-1) * valid).sum() / valid.sum())),
        "logit_spread": float((spread * w).sum() / wsum),
        "logit_absmax": float((big.masked_fill(~kmask.bool().expand_as(big), 0)
                               .abs().amax())),
        "attn_entropy": float((ent_n * w).sum() / wsum),
    }


def main():
    rng = np.random.RandomState(1234)
    n = 4000
    ids = np.zeros((n, 13), dtype=np.int64)
    for i in range(n):
        s = rng.randint(5, 14)          # >=5 cards so entropy is meaningful
        ids[i, :s] = rng.choice(np.arange(1, 53), size=s, replace=False)
    t_ids = torch.from_numpy(ids)
    pad = t_ids == 0

    print(f"n={n} hands (size U{{5..13}}); entropy normalised: 1.000 = 均勻(塌), 0.000 = one-hot\n")
    print(f"  {'arm':10s} {'tokRMS':>8s} {'tok‖x‖':>8s} {'logit幅寬':>10s} "
          f"{'|logit|max':>11s} {'attn熵':>8s}")

    rows = {}
    for name in ["plain", "featref", "B", "C", "D", "BCD"]:
        net = FACTORY[name]().to(DEV)
        sd = torch.load(f"{CKPT}/value_ab_{name}.pt", map_location=DEV)
        net.load_state_dict(sd["model"]); net.eval()
        s = attn_stats(net, t_ids, pad)
        rows[name] = s
        print(f"  {name:10s} {s['tok_rms']:8.3f} {s['tok_norm']:8.3f} "
              f"{s['logit_spread']:10.4f} {s['logit_absmax']:11.3f} {s['attn_entropy']:8.4f}")

    # untrained references (same init path) — what "healthy" looks like before training
    print()
    for name, tag in [("plain", "init-plain"), ("C", "init-C")]:
        torch.manual_seed(0)
        net = FACTORY[name]().to(DEV); net.eval()
        s = attn_stats(net, t_ids, pad)
        print(f"  {tag:10s} {s['tok_rms']:8.3f} {s['tok_norm']:8.3f} "
              f"{s['logit_spread']:10.4f} {s['logit_absmax']:11.3f} {s['attn_entropy']:8.4f}")

    p, c = rows["plain"], rows["C"]
    print(f"\n  C/plain: token‖x‖ ×{c['tok_norm']/p['tok_norm']:.1f}, "
          f"logit幅寬 ×{c['logit_spread']/p['logit_spread']:.1f}, "
          f"(norm比)² = ×{(c['tok_norm']/p['tok_norm'])**2:.1f}  ← 平方律預測 vs 實測")


if __name__ == "__main__":
    main()
