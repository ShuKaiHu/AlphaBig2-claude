"""BeliefNet + GRU history context (see belief_history_dataset.py for the "why"
and the exact per-step schema -- raw required/played card multi-hots, no
derived hand-type/strength features).

Same card-aware encoder + dot-product output head as ppo.belief_model.BeliefNet,
but the state vector also gets a GRU context over the full play history. The
motivation (a policy/belief net needs history to infer hidden information) is
the same one that justified the old (superseded) engine/model.py Big2Net's own
GRU-over-actionHistory design, but the per-step ENCODING here is new and
independent -- see belief_history_dataset.py's HIST_STEP_DIM_V2 schema.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.belief_history_dataset import HIST_STEP_DIM_V2
from ppo.belief_model import unseen_mask_from_obs  # noqa: F401 (re-exported for callers)
from ppo.network_cardaware import _to_batch

D = 64    # card embedding dim (matches belief_model.BeliefNet)
E = 256   # state / query dim
GRU_HIDDEN = 128


class BeliefNetHistory(nn.Module):
    def __init__(self, gru_hidden: int = GRU_HIDDEN, hist_dim: int = HIST_STEP_DIM_V2,
                 dropout: float = 0.0):
        """hist_dim defaults to the current V2 (raw-card) schema; pass the old
        V1 dim (29) explicitly when loading a checkpoint trained before the
        redesign -- see eval_belief_fair.py's before/after comparison.

        dropout: FUNCTIONAL dropout on the GRU context and encoder output (applied
        in forward, no module -> no state_dict keys -> old checkpoints load
        unchanged, and dropout=0 in eval() is an exact no-op = identical to the
        pre-regularization model)."""
        super().__init__()
        self.drop_p = float(dropout)
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.hand_attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        self.gru = nn.GRU(input_size=hist_dim, hidden_size=gru_hidden,
                           num_layers=1, batch_first=True)
        state_in = D + D + 3 * D + D + 3 + 1 + gru_hidden  # hand,seen,3xopp,trick,counts,passc,history
        self.enc = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        self.opp_query = nn.Linear(E, 3 * E)
        self.card_key = nn.Linear(D, E)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, obs, history_seq):
        hand_ids = obs["hand_ids"]
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn, _ = self.hand_attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        cards = self.card_emb.weight[1:]                       # (52,D)
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)
        trick_vec = obs["trick"] @ cards

        _, hn = self.gru(history_seq)          # hn: (1,B,gru_hidden)
        context = hn.squeeze(0)                # (B,gru_hidden)
        context = F.dropout(context, self.drop_p, self.training)

        si = torch.cat([hand_vec, seen_vec, opp_vec, trick_vec,
                         obs["counts"], obs["passc"], context], dim=-1)
        S = self.enc(si)                                       # (B,E)
        S = F.dropout(S, self.drop_p, self.training)
        oq = self.opp_query(S).reshape(-1, 3, E)
        ck = self.card_key(cards)                               # (52,E)
        logits = torch.einsum("bie,ce->bic", oq, ck) / math.sqrt(E) + self.bias
        return logits


def belief_history_batch(records, device):
    obs = _to_batch([r["obs"] for r in records], device)
    B = len(records)
    bel = np.zeros((B, 3, 52), dtype=np.float32)
    bmask = np.zeros((B, 52), dtype=np.float32)
    hist_len, hist_dim = records[0]["history"].shape  # infer dim from data (V1=29 or V2=108)
    hist = np.zeros((B, hist_len, hist_dim), dtype=np.float32)
    for i, r in enumerate(records):
        bel[i] = r["belief"]; bmask[i] = r["bmask"]; hist[i] = r["history"]
    return {"obs": obs,
            "belief": torch.from_numpy(bel).to(device),
            "bmask": torch.from_numpy(bmask).to(device),
            "history": torch.from_numpy(hist).to(device)}
