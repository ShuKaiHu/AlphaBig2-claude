"""ValueNetV2 — anti-collapse encoder variants for the public-info value net.

Motivation (measured on value_best.pt, 2026-08-04 probe session): the trained
hand path carries ~ZERO linearly-decodable structure (min_plays R2 0.000, hand
size 0.000, n_pairs ~0) — WORSE than its own random init (0.032). Mechanism:
card_emb row norms collapsed 7.9 -> 0.38 and hand_vec variance fell ~24x, so
attention logits (~ norm^2) went to zero and the softmax degenerated to uniform.
The shared embedding table serves two paths with OPPOSITE scale preferences:
the bag-sum path (seen = sum of up to 52 rows) wants small rows, the attention
path (q.k scales with norm^2) wants large ones — and the value signal is too
starved to make the hand path fight back (see workflow wf_bdecf93d probe).

Three independent levers, all "give the model room to learn on its own" — NO
handcrafted-feature supervision is injected anywhere (deliberate: min_plays is
a greedy upper bound, not a dominating factor; baking it in as a loss would
bias the representation toward one heuristic):

  B  split_emb : separate hand_emb (attention tokens) / bag_emb (seen/opp/trick
                 basis) — removes the scale tug-of-war entirely.
  C  pre_ln    : LayerNorm on hand tokens BEFORE attention — attention sharpness
                 becomes norm-independent (standard pre-LN), so even a shared
                 table cannot flatten the softmax.
     bag_mean  : seen/opp/trick bags divide by their card count — kills the
                 bag path's small-norm pressure at the source.
  D  residual  : tokens = h + attn(h) — preserves the linear bag code through
                 the attention (measured: attention alone degrades #2s-count
                 decodability 0.999 -> 0.48 because nothing carries h forward).
     own_count : own hand size / 13 appended to the state concat — mean-pool
                 divides size away and obs counts are OPPONENTS-ONLY, so the
                 model currently has no direct access to its own count.

All-off == architecture of value_model.ValueNet (kept as the baseline arm in
train_value_collapse_ab.py; this class is not meant to replace it until the
pre-registered A/B rules there say so).
"""
import bootstrap  # noqa: F401

import torch
import torch.nn as nn

from ppo.network_cardaware import _to_batch  # noqa: F401 (re-export for callers)
from value_model import SCALE, D, E


class ValueNetV2(nn.Module):
    def __init__(self, extra_dim=0, split_emb=False, pre_ln=False, bag_mean=False,
                 residual=False, own_count=False):
        super().__init__()
        self.extra_dim = extra_dim
        self.split_emb = split_emb
        self.pre_ln = pre_ln
        self.bag_mean = bag_mean
        self.residual = residual
        self.own_count = own_count

        self.hand_emb = nn.Embedding(53, D, padding_idx=0)
        self.bag_emb = nn.Embedding(53, D, padding_idx=0) if split_emb else self.hand_emb
        self.tok_ln = nn.LayerNorm(D) if pre_ln else None
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)

        state_in = D + D + 3 * D + D + 3 + 1 + (1 if own_count else 0) + extra_dim
        self.enc = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        self.value_head = nn.Sequential(
            nn.Linear(E, E // 2), nn.ReLU(),
            nn.Linear(E // 2, 1), nn.Tanh(),
        )

    def hand_vec(self, hand_ids):
        """(B,13) long -> (B,D). Split out so the collapse probe can read the
        hand path in isolation (same reason the 2026-08 probes existed)."""
        pad = hand_ids == 0
        h = self.hand_emb(hand_ids)
        t = self.tok_ln(h) if self.tok_ln is not None else h
        attn, _ = self.hand_attn(t, t, t, key_padding_mask=pad, need_weights=False)
        tok = h + attn if self.residual else attn
        keep = (~pad).float().unsqueeze(-1)
        return (tok * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def _bag(self, multihot, cards):
        v = multihot @ cards
        if self.bag_mean:
            v = v / multihot.sum(-1, keepdim=True).clamp(min=1.0)
        return v

    def forward(self, obs):
        hand_ids = obs["hand_ids"]
        hv = self.hand_vec(hand_ids)
        cards = self.bag_emb.weight[1:]                        # (52,D)
        seen_vec = self._bag(obs["seen"], cards)
        opp_vec = self._bag(obs["opp"], cards).reshape(obs["opp"].shape[0], -1)
        trick_vec = self._bag(obs["trick"], cards)
        parts = [hv, seen_vec, opp_vec, trick_vec, obs["counts"], obs["passc"]]
        if self.own_count:
            own = (hand_ids != 0).float().sum(-1, keepdim=True) / 13.0
            parts.append(own)
        if self.extra_dim:
            parts.append(obs["feat"])
        si = torch.cat(parts, dim=-1)
        return self.value_head(self.enc(si)).squeeze(-1)       # (B,) in [-1,1]

    @torch.no_grad()
    def value_from_obs(self, st, device="cpu"):
        v = self.forward(_to_batch([st], device))[0].item()
        return v * SCALE
