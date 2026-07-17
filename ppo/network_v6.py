"""V6: bigger card-aware net + opponent-hand BELIEF head (belief-conditioned policy).

Idea (imperfect-information): from PUBLIC info only (our hand, played cards,
opponent counts, trick), predict each opponent's hidden hand (belief), then let
the policy/value condition on that prediction. Belief is trained toward an ORACLE
target (opponents' actual hidden cards) — available in self-play and in the
showdown-reconstructed online dataset. The ORACLE is used ONLY for the belief
loss; it is NEVER fed into the observation (no info leak — works online where we
must predict, not peek).

Shapes: belief = (3, 52) = P(opponent o holds card c), opponents clockwise from us.
"""
import math

import numpy as np
import torch
import torch.nn as nn

from ppo.action_features import ACTION_FEATURES, AF, PASS_IDX
from ppo.network_cardaware import obs_from_env, _to_batch  # same observation as cardaware

HAND_PAD = 13
D = 64       # card embedding dim (was 32)
E = 256      # state/action embedding dim (was 128)
BELIEF = 3 * 52


def belief_target_from_env(env):
    """Oracle (3,52): opponents' ACTUAL current hands, clockwise from us.
    For the belief LOSS only — needs the full game state (self-play / reconstruction)."""
    g = env.game
    me = env.current_player
    order = [((me - 1 + k) % 4) + 1 for k in (1, 2, 3)]
    tgt = np.zeros((3, 52), dtype=np.float32)
    for i, p in enumerate(order):
        for c in g.currentHands[p]:
            c = int(c)
            if 1 <= c <= 52:
                tgt[i, c - 1] = 1.0
    return tgt


def unseen_mask_from_obs(obs):
    """(52,) 1 for cards that COULD be in an opponent's hand (not ours, not played).
    Belief loss is applied only over these — the rest are known."""
    own = set(int(c) for c in obs["hand_ids"] if c > 0)
    seen = set(int(i) for i in np.flatnonzero(obs["seen"]))  # 0-based card-1
    m = np.ones(52, dtype=np.float32)
    for c in own:
        m[c - 1] = 0.0
    for i in seen:
        m[i] = 0.0
    return m


class CardAwareV6(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_emb = nn.Embedding(53, D, padding_idx=0)
        self.attn = nn.MultiheadAttention(D, num_heads=4, batch_first=True)
        state_in = D + D + D + 3 * D + 3 + 1
        self.state_mlp = nn.Sequential(
            nn.Linear(state_in, E), nn.ReLU(),
            nn.Linear(E, E), nn.ReLU(), nn.LayerNorm(E),
        )
        # belief head: state -> per-opponent per-card logits
        self.belief_head = nn.Sequential(nn.Linear(E, E), nn.ReLU(), nn.Linear(E, BELIEF))
        # fuse belief back into the state (belief-conditioned policy/value)
        self.belief_feat = nn.Sequential(nn.Linear(BELIEF, D), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(E + D, E), nn.ReLU(), nn.LayerNorm(E))
        # action scoring + value (on belief-conditioned state)
        self.act_mlp = nn.Sequential(nn.Linear(AF, E), nn.ReLU(), nn.Linear(E, E))
        self.s_proj = nn.Linear(E, E)
        self.value_head = nn.Sequential(nn.Linear(E, E), nn.ReLU(), nn.Linear(E, 1))

    def state_embedding(self, obs):
        hand_ids = obs["hand_ids"]
        pad = hand_ids == 0
        h = self.card_emb(hand_ids)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=pad)
        keep = (~pad).float().unsqueeze(-1)
        hand_vec = (attn_out * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        cards = self.card_emb.weight[1:]
        trick_vec = obs["trick"] @ cards
        seen_vec = obs["seen"] @ cards
        opp_vec = (obs["opp"] @ cards).reshape(obs["opp"].shape[0], -1)
        si = torch.cat([hand_vec, trick_vec, seen_vec, opp_vec, obs["counts"], obs["passc"]], dim=-1)
        return self.state_mlp(si)

    def forward(self, obs, act_feats, act_mask):
        S = self.state_embedding(obs)                       # (B,E)
        belief_logits = self.belief_head(S).reshape(-1, 3, 52)
        belief = torch.sigmoid(belief_logits)               # (B,3,52)
        # condition policy/value on the (detached) belief prediction
        S2 = self.fuse(torch.cat([S, self.belief_feat(belief.detach().reshape(belief.shape[0], -1))], dim=-1))
        A = self.act_mlp(act_feats)
        score = (A * self.s_proj(S2).unsqueeze(1)).sum(-1) / math.sqrt(E)
        score = score.masked_fill(~act_mask, -1e9)
        value = self.value_head(S2).squeeze(-1)
        return score, value, belief_logits

    @torch.no_grad()
    def greedy(self, env, device):
        from ppo.network_cardaware import legal_action_data
        legal_idx, feats = legal_action_data(env)
        obs = _to_batch([obs_from_env(env)], device)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, _, _ = self.forward(obs, af, am)
        return int(legal_idx[int(torch.argmax(logits[0]).item())])

    @torch.no_grad()
    def act(self, env, device):
        """Stochastic sample -- needed so PPO_V6 can be used as a pool
        opponent in ppo_trainer.py's collect() (opponents act stochastically
        too, not just greedily). Mirrors CardAwareActorCritic.act; drops the
        belief_logits output (not needed for action selection)."""
        from ppo.network_cardaware import legal_action_data
        legal_idx, feats = legal_action_data(env)
        obs_np = obs_from_env(env)
        obs = _to_batch([obs_np], device)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        logits, value, _ = self.forward(obs, af, am)
        dist = torch.distributions.Categorical(logits=logits[0])
        pos = dist.sample()
        action = int(legal_idx[int(pos.item())])
        rec = {"obs": obs_np, "feats": feats, "pos": int(pos.item())}
        return action, float(dist.log_prob(pos).item()), float(value.item()), rec


def evaluate_batch_v6(net, batch):
    """PPO update pass for CardAwareV6 as a LEARNER (not just a frozen pool
    opponent) -- reuses network_cardaware.build_batch's plain {obs,act_feats,
    act_mask,pos} schema (no belief/bmask fields; those are only needed for
    the separate BC+belief training loss in build_batch_v6/belief_loss below,
    a different training mode from PPO). Discards belief_logits -- the PPO
    objective here is plain policy+value, matching evaluate_batch in
    network_cardaware.py/network_cardaware_history.py."""
    logits, value, _ = net.forward(batch["obs"], batch["act_feats"], batch["act_mask"])
    dist = torch.distributions.Categorical(logits=logits)
    pos = batch["pos"]
    return dist.log_prob(pos), dist.entropy(), value


def build_batch_v6(records, device):
    """records: {obs, feats(L,AF), pos, belief(3,52), bmask(52)}."""
    obs = _to_batch([r["obs"] for r in records], device)
    Lmax = max(r["feats"].shape[0] for r in records)
    B = len(records)
    af = np.zeros((B, Lmax, AF), dtype=np.float32)
    am = np.zeros((B, Lmax), dtype=bool)
    pos = np.zeros(B, dtype=np.int64)
    bel = np.zeros((B, 3, 52), dtype=np.float32)
    bmask = np.zeros((B, 52), dtype=np.float32)
    for i, r in enumerate(records):
        L = r["feats"].shape[0]
        af[i, :L] = r["feats"]; am[i, :L] = True; pos[i] = r["pos"]
        bel[i] = r["belief"]; bmask[i] = r["bmask"]
    return {"obs": obs,
            "act_feats": torch.from_numpy(af).to(device),
            "act_mask": torch.from_numpy(am).to(device),
            "pos": torch.from_numpy(pos).to(device),
            "belief": torch.from_numpy(bel).to(device),
            "bmask": torch.from_numpy(bmask).to(device)}


def belief_loss(belief_logits, target, bmask):
    """BCE over unseen cards only. bmask (B,52) -> broadcast over 3 opponents."""
    per = nn.functional.binary_cross_entropy_with_logits(belief_logits, target, reduction="none")
    m = bmask.unsqueeze(1)  # (B,1,52)
    return (per * m).sum() / m.expand_as(per).sum().clamp(min=1.0)
