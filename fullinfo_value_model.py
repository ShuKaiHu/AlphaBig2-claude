"""FULL-INFO value model — the world-aware ISMCTS leaf.

Unlike VALUE_minplays (public obs only), this eats the DETERMINIZED world: it also
sees each opponent's REMAINING hand (through the same card embedding), so it can
tell a losing world (opp holds a bomb) from a winning one instead of averaging over
all hidden hands. Used as the ISMCTS leaf, where determinization already provides
the opponents' cards.

obs = public obs (obs_from_env) + opp_hands (3x52 remaining, clockwise from me)
      + feat (4 min_plays: me + 3 opps — the proven strength feature, now full-info).
"""
import bootstrap  # noqa: F401

import numpy as np
import torch
import torch.nn as nn

from ppo.network_cardaware import obs_from_env, _to_batch
from value_model import SCALE, D, E
from hand_features import min_plays_to_empty


def fullinfo_obs_from_env(env):
    """Public obs + opponents' remaining hands + 4 min_plays features."""
    d = obs_from_env(env)
    g = env.game
    me = env.current_player
    order = [((me - 1 + k) % 4) + 1 for k in (1, 2, 3)]     # clockwise from me (matches obs['opp'])
    oh = np.zeros((3, 52), dtype=np.float32)
    for i, p in enumerate(order):
        for c in g.currentHands[p]:
            oh[i][int(c) - 1] = 1.0
    d["opp_hands"] = oh
    mp = [min_plays_to_empty(list(g.currentHands[me])) / 13.0]
    mp += [min_plays_to_empty(list(g.currentHands[p])) / 13.0 for p in order]
    d["feat"] = np.array(mp, dtype=np.float32)              # [me, opp1, opp2, opp3]
    return d


def fullinfo_batch(obs_list, device):
    b = _to_batch(obs_list, device)
    b["opp_hands"] = torch.tensor(np.stack([o["opp_hands"] for o in obs_list]),
                                  dtype=torch.float32, device=device)
    b["feat"] = torch.tensor(np.stack([o["feat"] for o in obs_list]),
                             dtype=torch.float32, device=device)
    return b


class FullInfoValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        # hand + seen + 3*opp(played) + 3*opp_hands(remaining) + trick + counts(3) + passc(1) + feat(4)
        state_in = D + D + 3 * D + 3 * D + D + 3 + 1 + 4
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
        cards = self.card_emb.weight[1:]                    # (52, D)
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)
        oh_vec = (obs["opp_hands"] @ cards).reshape(obs["opp_hands"].shape[0], -1)
        trick_vec = obs["trick"] @ cards
        si = torch.cat([hand_vec, seen_vec, opp_vec, oh_vec, trick_vec,
                        obs["counts"], obs["passc"], obs["feat"]], dim=-1)
        return self.value_head(self.enc(si)).squeeze(-1)    # (B,) in [-1,1]

    @torch.no_grad()
    def value_from_env(self, env, device="cpu"):
        return self.forward(fullinfo_batch([fullinfo_obs_from_env(env)], device))[0].item() * SCALE
