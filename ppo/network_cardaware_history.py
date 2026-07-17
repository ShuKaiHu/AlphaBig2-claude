"""CardAwareActorCritic + GRU history context -- same idea and same per-step
schema as ppo/belief_model_history.py::BeliefNetHistory (raw required/played
card multi-hots per step, no derived hand-type/strength features), applied to
the POLICY instead of belief. See belief_history_dataset.py for the "why".

Checkpoint selection for training this must use REAL self-play gameplay
(evaluate_vs_all in ppo/eval_baselines.py), never a decision-level train/val
accuracy split -- that leaky-split mistake is exactly what inflated the old
belief numbers this session; ppo/train_bc.py already gets this right for the
plain (no-history) policy, and train_policy_history.py must match it.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from ppo.action_features import ACTION_FEATURES, AF
from ppo.belief_history_dataset import HIST_STEP_DIM_V2, HISTORY_LEN_V2, encode_history_steps_v2
from ppo.network_cardaware import D, E, _to_batch, legal_action_data, obs_from_env

GRU_HIDDEN = 128


class CardAwareActorCriticHistory(nn.Module):
    def __init__(self, gru_hidden: int = GRU_HIDDEN, hist_dim: int = HIST_STEP_DIM_V2):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        self.gru = nn.GRU(input_size=hist_dim, hidden_size=gru_hidden,
                           num_layers=1, batch_first=True)
        state_in = D + D + D + 3 * D + 3 + 1 + gru_hidden  # hand,trick,seen,3*opp,counts,passc,history
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(),
            nn.LayerNorm(E),
        )
        self.act_mlp = nn.Sequential(
            nn.Linear(AF, E), nn.ReLU(),
            nn.Linear(E, E),
        )
        self.s_proj = nn.Linear(E, E)
        self.value_head = nn.Sequential(nn.Linear(E, E), nn.ReLU(), nn.Linear(E, 1))

    def state_embedding(self, obs, history_seq):
        hand_ids = obs["hand_ids"]
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn_out * keep).sum(1) / keep.sum(1).clamp(min=1.0)

        cards = self.card_emb.weight[1:]
        trick_vec = obs["trick"] @ cards
        seen_vec = obs["seen"] @ cards
        opp_vec = obs["opp"] @ cards
        opp_vec = opp_vec.reshape(opp_vec.shape[0], -1)

        _, hn = self.gru(history_seq)          # hn: (1,B,gru_hidden)
        context = hn.squeeze(0)                # (B,gru_hidden)

        state_in = torch.cat([hand_vec, trick_vec, seen_vec, opp_vec,
                              obs["counts"], obs["passc"], context], dim=-1)
        return self.state_mlp(state_in)

    def forward(self, obs, history_seq, act_feats, act_mask):
        S = self.state_embedding(obs, history_seq)
        A = self.act_mlp(act_feats)
        q = self.s_proj(S).unsqueeze(1)
        score = (A * q).sum(-1) / math.sqrt(E)
        score = score.masked_fill(~act_mask, -1e9)
        value = self.value_head(S).squeeze(-1)
        return score, value

    @torch.no_grad()
    def greedy(self, env, device):
        obs = _to_batch([obs_from_env(env)], device)
        hist = live_history_tensor(env, device)
        legal_idx, feats = legal_action_data(env)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, _ = self.forward(obs, hist, af, am)
        return int(legal_idx[int(torch.argmax(logits[0]).item())])

    @torch.no_grad()
    def act(self, env, device):
        """Stochastic sample for PPO rollout collection (mirrors
        network_cardaware.CardAwareActorCritic.act) -- greedy() above is
        argmax-only and was never meant for training-time exploration."""
        obs_np = obs_from_env(env)
        obs = _to_batch([obs_np], device)
        hist_np = live_history_tensor(env, "cpu")[0].numpy()
        hist = torch.from_numpy(hist_np).unsqueeze(0).to(device)
        legal_idx, feats = legal_action_data(env)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, value = self.forward(obs, hist, af, am)
        dist = torch.distributions.Categorical(logits=logits[0])
        pos = dist.sample()
        action = int(legal_idx[int(pos.item())])
        rec = {"obs": obs_np, "feats": feats, "pos": int(pos.item()), "history": hist_np}
        return action, float(dist.log_prob(pos).item()), float(value.item()), rec


def steps_from_action_history(action_history):
    """Convert the LIVE engine's big2Game.actionHistory (list of dicts with
    player/hand/pass/forced_skip -- see big2Game.py's updateGame()) into
    (seat, required_cards, played_cards) tuples, using the exact same
    trick-tracking rule as belief_history_dataset.py's corpus-reconstruction
    path (a trick resets after 3 consecutive passes). forced_skip entries are
    engine bookkeeping (auto-skipping an already-passed player mid-trick), not
    real decisions -- excluded, same as the corpus path (which has none)."""
    steps = []
    trick_cards, passed = None, set()
    for entry in action_history:
        if entry.get("forced_skip"):
            continue
        seat = entry["player"] - 1  # engine players are 1-indexed
        is_pass = entry["pass"]
        cards = [] if is_pass else [int(c) for c in entry["hand"]]
        required_cards = [] if trick_cards is None else list(trick_cards)
        played_cards = [] if is_pass else cards
        steps.append((seat, required_cards, played_cards))
        if is_pass:
            passed.add(seat)
            if len(passed) >= 3:
                trick_cards, passed = None, set()
        else:
            trick_cards, passed = cards, set()
    return steps


def live_history_tensor(env, device, history_len=HISTORY_LEN_V2):
    """(1, history_len, HIST_STEP_DIM_V2) tensor built from the env's true,
    already-maintained actionHistory -- correct regardless of which policy
    (ours or an opponent's) drove each past action."""
    steps = steps_from_action_history(env.game.actionHistory)
    hist = encode_history_steps_v2(steps, history_len=history_len)
    return torch.from_numpy(hist).unsqueeze(0).to(device)


def build_batch_history(records, device):
    """records: {obs, feats(L,AF), pos, history(history_len,HIST_STEP_DIM_V2)}."""
    obs = _to_batch([r["obs"] for r in records], device)
    Lmax = max(r["feats"].shape[0] for r in records)
    B = len(records)
    af = np.zeros((B, Lmax, AF), dtype=np.float32)
    am = np.zeros((B, Lmax), dtype=bool)
    pos = np.zeros(B, dtype=np.int64)
    hist_len, hist_dim = records[0]["history"].shape
    hist = np.zeros((B, hist_len, hist_dim), dtype=np.float32)
    for i, r in enumerate(records):
        L = r["feats"].shape[0]
        af[i, :L] = r["feats"]
        am[i, :L] = True
        pos[i] = r["pos"]
        hist[i] = r["history"]
    return {
        "obs": obs,
        "act_feats": torch.from_numpy(af).to(device),
        "act_mask": torch.from_numpy(am).to(device),
        "pos": torch.from_numpy(pos).to(device),
        "history": torch.from_numpy(hist).to(device),
    }


def evaluate_batch(net, batch):
    """PPO update pass -- mirrors network_cardaware.evaluate_batch, plus the
    history_seq argument this architecture needs."""
    logits, value = net.forward(batch["obs"], batch["history"], batch["act_feats"], batch["act_mask"])
    dist = torch.distributions.Categorical(logits=logits)
    pos = batch["pos"]
    return dist.log_prob(pos), dist.entropy(), value
