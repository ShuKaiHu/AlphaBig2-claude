"""Dedicated opponent-hand BELIEF model (standalone, for grafting onto search).

Given ONE player's public view (own hand, all played cards, per-opponent played
cards, counts, current trick, pass count) it predicts P(opponent i holds card c)
for the 3 opponents — a (3,52) matrix. Trained purely on the showdown-reconstructed
online games (oracle = opponents' actual hidden hands; loss-only, never in input).

Unlike the V6 belief head (a side-task of the policy net), this net's encoder is
specialized 100% for belief, and the output is card-aware: logit[i][c] =
opp_query[i] · card_key[c], so structure is shared across cards. Meant to be
dropped into a determinized search (importance-sample worlds from this belief) —
e.g. the AlphaZero / full-info-value-net line.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from ppo.network_cardaware import obs_from_env, _to_batch  # same public observation

D = 64    # card embedding dim
E = 256   # state / query dim


def unseen_mask_from_obs(obs):
    """(52,) 1 for cards that COULD be an opponent's (not ours, not played)."""
    own = set(int(c) for c in obs["hand_ids"] if c > 0)
    seen = set(int(i) for i in np.flatnonzero(obs["seen"]))
    m = np.ones(52, dtype=np.float32)
    for c in own:
        m[c - 1] = 0.0
    for i in seen:
        m[i] = 0.0
    return m


class BeliefNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        state_in = D + D + 3 * D + D + 3 + 1  # hand, seen, 3×opp-played, trick, counts, passc
        self.enc = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        self.opp_query = nn.Linear(E, 3 * E)   # one query vector per opponent
        self.card_key = nn.Linear(D, E)        # card -> key (card-aware output)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        hand_ids = obs["hand_ids"]
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn, _ = self.hand_attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        cards = self.card_emb.weight[1:]                       # (52,D)
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)  # (B,3D)
        trick_vec = obs["trick"] @ cards
        si = torch.cat([hand_vec, seen_vec, opp_vec, trick_vec, obs["counts"], obs["passc"]], dim=-1)
        S = self.enc(si)                                       # (B,E)
        oq = self.opp_query(S).reshape(-1, 3, E)              # (B,3,E)
        ck = self.card_key(cards)                              # (52,E)
        logits = torch.einsum("bie,ce->bic", oq, ck) / math.sqrt(E) + self.bias  # (B,3,52)
        return logits


def belief_batch(records, device):
    obs = _to_batch([r["obs"] for r in records], device)
    B = len(records)
    bel = np.zeros((B, 3, 52), dtype=np.float32)
    bmask = np.zeros((B, 52), dtype=np.float32)
    for i, r in enumerate(records):
        bel[i] = r["belief"]; bmask[i] = r["bmask"]
    return {"obs": obs,
            "belief": torch.from_numpy(bel).to(device),
            "bmask": torch.from_numpy(bmask).to(device)}


def belief_bce(logits, target, bmask):
    per = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    m = bmask.unsqueeze(1)
    return (per * m).sum() / m.expand_as(per).sum().clamp(min=1.0)


@torch.no_grad()
def belief_from_obs(net, st, device="cpu"):
    """(3,52) P(opp i holds card c), masked to unseen — for search/determinization."""
    logits = net(_to_batch([st], device))
    return (torch.sigmoid(logits)[0].cpu().numpy()) * unseen_mask_from_obs(st)
