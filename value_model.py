"""Imperfect-information VALUE model: public state -> expected final score.

Input  = the acting player's PUBLIC view (own hand, all played cards, per-opponent
         played cards, counts, current trick, pass count) — the cardaware obs.
Output = a scalar in [-1, 1] = tanh(final score / SCALE) the acting player is
         expected to end the game with, averaging over the hidden hands consistent
         with the public state AND the self-play policy's continuation.

This is the AlphaGo value-net analogue for an imperfect-information game: V(public).
No determinization needed at use time (unlike the full-info Big2ValueNet) — it eats
public info directly. SCALE matches the engine reward scaling used elsewhere (13).
"""
import bootstrap  # noqa: F401  (sets sys.path + chdir before engine imports)

import math
import numpy as np
import torch
import torch.nn as nn

from ppo.network_cardaware import obs_from_env, _to_batch  # public obs builder

SCALE = 13.0
D = 64    # card embedding dim
E = 256   # hidden dim


class ValueNet(nn.Module):
    def __init__(self, extra_dim=0):
        super().__init__()
        self.extra_dim = extra_dim   # hand-strength features (dominance probs + min-plays)
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        state_in = D + D + 3 * D + D + 3 + 1 + extra_dim  # +obs["feat"] if extra_dim
        self.enc = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        self.value_head = nn.Sequential(
            nn.Linear(E, E // 2), nn.ReLU(),
            nn.Linear(E // 2, 1), nn.Tanh(),
        )

    def forward(self, obs):
        hand_ids = obs["hand_ids"]
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn, _ = self.hand_attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        cards = self.card_emb.weight[1:]
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)
        trick_vec = obs["trick"] @ cards
        si = torch.cat([hand_vec, seen_vec, opp_vec, trick_vec, obs["counts"], obs["passc"]], dim=-1)
        if self.extra_dim:
            si = torch.cat([si, obs["feat"]], dim=-1)
        return self.value_head(self.enc(si)).squeeze(-1)  # (B,) in [-1,1]

    @torch.no_grad()
    def value_from_obs(self, st, device="cpu"):
        """Expected final score (REAL units) for the acting player at public state st."""
        v = self.forward(_to_batch([st], device))[0].item()
        return v * SCALE


def build_value_batch(records, device):
    obs = _to_batch([r["obs"] for r in records], device)
    tgt = torch.tensor(np.array([r["target"] for r in records], dtype=np.float32), device=device)
    if "feat" in records[0]:
        obs["feat"] = torch.tensor(np.stack([r["feat"] for r in records]).astype(np.float32), device=device)
    return {"obs": obs, "target": tgt}
