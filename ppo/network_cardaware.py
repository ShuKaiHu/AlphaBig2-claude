"""Card-aware actor-critic with dot-product action scoring (paper architecture).

State side ("card-aware"):
  * a shared card embedding table (idx 0 = pad, 1..52 = cards)
  * the acting hand is embedded, run through masked self-attention so cards
    interact, then masked-mean-pooled into a hand vector
  * the trick / seen / each opponent's played cards are 52-indicators turned into
    vectors via the SAME embedding table (sum of present-card embeddings)
  * concatenated with opponent counts + pass count -> MLP -> state embedding S

Action side ("dot-product"):
  * each LEGAL action's precomputed feature vector (action_features.AF) -> MLP
    -> action embedding A_i
  * score_i = (proj(S) · A_i) / sqrt(E)  (scaled dot product), masked to legal
  * softmax(scores) = policy; a separate MLP on S gives V(S)

Only legal actions are scored, and similar actions share learning — the whole
point versus a fixed 14739-way head.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from ppo.action_features import ACTION_FEATURES, AF, PASS_IDX

HAND_PAD = 13
D = 32      # card embedding dim
E = 128     # state / action embedding dim


# ── obs extraction (numpy, single env) ──────────────────────────────────────
def obs_from_env(env):
    g = env.game
    me = env.current_player
    hand = [int(c) for c in g.currentHands[me]]
    hand_ids = np.zeros(HAND_PAD, dtype=np.int64)
    hand_ids[:len(hand)] = hand[:HAND_PAD]

    trick = np.zeros(52, dtype=np.float32)
    if g.control == 0:
        for c in g.handsPlayed[g.goIndex - 1].hand:
            trick[int(c) - 1] = 1.0

    seen = (g.cardsPlayed != 0).any(axis=0).astype(np.float32)  # (52,)

    opp = np.zeros((3, 52), dtype=np.float32)
    counts = np.zeros(3, dtype=np.float32)
    order = [((me - 1 + k) % 4) + 1 for k in (1, 2, 3)]  # clockwise from me
    for i, p in enumerate(order):
        opp[i] = (g.cardsPlayed[p - 1] != 0).astype(np.float32)
        counts[i] = g.currentHands[p].size / 13.0
    passc = np.array([g.passCount / 3.0], dtype=np.float32)
    return {"hand_ids": hand_ids, "trick": trick, "seen": seen,
            "opp": opp, "counts": counts, "passc": passc}


def legal_action_data(env):
    """(legal_idx int array, feats (L,AF) float array) for the current state."""
    mask = env.get_valid_actions()
    legal_idx = np.flatnonzero(mask).astype(np.int64)
    feats = ACTION_FEATURES[legal_idx]
    return legal_idx, feats


# ── network ─────────────────────────────────────────────────────────────────
class CardAwareActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        state_in = D + D + D + 3 * D + 3 + 1  # hand,trick,seen,3*opp,counts,passc
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

    def state_embedding(self, obs):
        hand_ids = obs["hand_ids"]                      # (B,13) long
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)                     # (B,13,D)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)             # (B,13,1)
        hand_vec = (attn_out * keep).sum(1) / keep.sum(1).clamp(min=1.0)

        cards = self.card_emb.weight[1:]                # (52,D)
        trick_vec = obs["trick"] @ cards                # (B,D)
        seen_vec = obs["seen"] @ cards
        opp_vec = obs["opp"] @ cards                    # (B,3,D)
        opp_vec = opp_vec.reshape(opp_vec.shape[0], -1)  # (B,3D)

        state_in = torch.cat([hand_vec, trick_vec, seen_vec, opp_vec,
                              obs["counts"], obs["passc"]], dim=-1)
        return self.state_mlp(state_in)                 # (B,E)

    def forward(self, obs, act_feats, act_mask):
        """obs dict of tensors; act_feats (B,L,AF); act_mask (B,L) bool.
        Returns logits (B,L) masked, value (B,)."""
        S = self.state_embedding(obs)                   # (B,E)
        A = self.act_mlp(act_feats)                     # (B,L,E)
        q = self.s_proj(S).unsqueeze(1)                 # (B,1,E)
        score = (A * q).sum(-1) / math.sqrt(E)          # (B,L)
        score = score.masked_fill(~act_mask, -1e9)
        value = self.value_head(S).squeeze(-1)          # (B,)
        return score, value

    # ── single-env helpers (collection / eval) ──────────────────────────────
    @torch.no_grad()
    def act(self, env, device):
        obs = _to_batch([obs_from_env(env)], device)
        legal_idx, feats = legal_action_data(env)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)         # (1,L,AF)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, value = self.forward(obs, af, am)
        dist = torch.distributions.Categorical(logits=logits[0])
        pos = dist.sample()
        action = int(legal_idx[int(pos.item())])
        rec = {"obs": obs_from_env(env), "feats": feats, "pos": int(pos.item())}
        return action, float(dist.log_prob(pos).item()), float(value.item()), rec

    @torch.no_grad()
    def greedy(self, env, device):
        obs = _to_batch([obs_from_env(env)], device)
        legal_idx, feats = legal_action_data(env)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, _ = self.forward(obs, af, am)
        return int(legal_idx[int(torch.argmax(logits[0]).item())])


# ── batching for the PPO update ──────────────────────────────────────────────
def _to_batch(obs_list, device):
    def stack(key, dtype):
        return torch.tensor(np.stack([o[key] for o in obs_list]), dtype=dtype, device=device)
    return {
        "hand_ids": stack("hand_ids", torch.long),
        "trick": stack("trick", torch.float32),
        "seen": stack("seen", torch.float32),
        "opp": stack("opp", torch.float32),
        "counts": stack("counts", torch.float32),
        "passc": stack("passc", torch.float32),
    }


def build_batch(records, device):
    """Pad variable-length legal-action sets to the minibatch max."""
    obs = _to_batch([r["obs"] for r in records], device)
    Lmax = max(r["feats"].shape[0] for r in records)
    B = len(records)
    af = np.zeros((B, Lmax, AF), dtype=np.float32)
    am = np.zeros((B, Lmax), dtype=bool)
    pos = np.zeros(B, dtype=np.int64)
    for i, r in enumerate(records):
        L = r["feats"].shape[0]
        af[i, :L] = r["feats"]
        am[i, :L] = True
        pos[i] = r["pos"]
    return {
        "obs": obs,
        "act_feats": torch.from_numpy(af).to(device),
        "act_mask": torch.from_numpy(am).to(device),
        "pos": torch.from_numpy(pos).to(device),
    }


def evaluate_batch(net, batch):
    logits, value = net.forward(batch["obs"], batch["act_feats"], batch["act_mask"])
    dist = torch.distributions.Categorical(logits=logits)
    pos = batch["pos"]
    return dist.log_prob(pos), dist.entropy(), value
