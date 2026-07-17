"""Dominance-probability model: public obs -> P(my best single/pair/full-house is nuts).

Same architecture family as the belief/value nets (card-emb + hand attention +
encoder), trained with BCE on the showdown-oracle 0/1 labels → calibrated soft
probabilities. Use the outputs as 'hand-strength' features for the value / MCTS leaf.
"""
import bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

from ppo.network_cardaware import obs_from_env, _to_batch  # noqa: F401 (obs_from_env re-exported)
from dominance_oracle import LABELS

D, E, NOUT = 64, 256, len(LABELS)


class DominanceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        state_in = D + D + 3 * D + D + 3 + 1
        self.enc = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        self.head = nn.Sequential(nn.Linear(E, E // 2), nn.ReLU(), nn.Linear(E // 2, NOUT))

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
        return self.head(self.enc(si))           # (B, NOUT) logits

    @torch.no_grad()
    def probs_from_obs(self, st, device="cpu"):
        return torch.sigmoid(self.forward(_to_batch([st], device)))[0].cpu().numpy()  # (NOUT,)


def build_dom_batch(records, device):
    obs = _to_batch([r["obs"] for r in records], device)
    tgt = torch.tensor(np.stack([r["target"] for r in records]), dtype=torch.float32, device=device)
    return {"obs": obs, "target": tgt}
