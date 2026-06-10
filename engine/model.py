import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.features import STATIC_DIM, GRU_HIDDEN, HIST_STEP_DIM, HISTORY_LEN, OPP_HANDS_DIM
import enumerateOptions

ACTION_SIZE = enumerateOptions.passInd + 1  # 14739


class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class Big2Net(nn.Module):
    """
    AlphaZero-style network for Big 2.

    Architecture:
        GRU(history_seq) → context vector
        concat(static_feat, context) → input_proj → 4 × ResBlock → trunk
        trunk → policy head → logits (ACTION_SIZE,)
        trunk → value head  → scalar in [-1, 1]

    Uses LayerNorm instead of BatchNorm so it works correctly
    with batch_size=1 during MCTS inference.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_res_blocks: int = 4,
        gru_hidden: int = GRU_HIDDEN,
        action_size: int = ACTION_SIZE,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=HIST_STEP_DIM,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        input_dim = STATIC_DIM + gru_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_res_blocks)]
        )
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_size),
        )
        # Value head — 4-dim, one expected (normalized) terminal reward per
        # ABSOLUTE player index (value[p-1] = expected reward of player p).
        # This makes the network a proper N-player value function instead of a
        # 2-player zero-sum scalar, so MCTS backup never has to assume one
        # player's gain equals another's loss.
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Tanh(),
        )
        # Belief auxiliary head: predict which cards each of 3 opponents holds
        # Output: 52 * 3 = 156 logits (BCEWithLogits), only used during training
        self.belief_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 52 * 3),
        )

    def forward(self, static_feat, history_seq, valid_mask=None):
        """
        Args:
            static_feat:  (B, STATIC_DIM)   float32
            history_seq:  (B, HISTORY_LEN, HIST_STEP_DIM) float32
            valid_mask:   (B, ACTION_SIZE)  float32, 1=valid, 0=invalid (optional)
        Returns:
            policy_logits: (B, ACTION_SIZE)  — invalid actions masked to -1e9
            value:         (B, 4)            — tanh per absolute player ∈ [-1, 1]
            belief_logits: (B, 156)          — raw logits for 3 opponents × 52 cards
        """
        _, h = self.gru(history_seq)          # h: (1, B, gru_hidden)
        context = h.squeeze(0)                # (B, gru_hidden)
        x = torch.cat([static_feat, context], dim=-1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        logits = self.policy_head(x)
        if valid_mask is not None:
            logits = logits + (1.0 - valid_mask) * (-1e9)
        value = self.value_head(x)
        belief_logits = self.belief_head(x)
        return logits, value, belief_logits

    @torch.no_grad()
    def predict(self, static_feat, history_seq, valid_mask=None):
        """
        Single-sample inference. Inputs are numpy arrays.
        Returns:
            probs: np.ndarray (ACTION_SIZE,)  softmax probabilities (valid actions only)
            value: np.ndarray (4,)  — expected normalized reward per absolute player
        """
        self.eval()
        sf = torch.FloatTensor(static_feat).unsqueeze(0)
        hs = torch.FloatTensor(history_seq).unsqueeze(0)
        vm = torch.FloatTensor(valid_mask).unsqueeze(0) if valid_mask is not None else None
        logits, val, _ = self.forward(sf, hs, vm)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        value = val.squeeze(0).cpu().numpy().astype(np.float32)  # (4,)
        return probs, value


class Big2ValueNet(nn.Module):
    """Full-information (god-view) value network — V9.

    Evaluates a position knowing ALL FOUR hands. Only ever used inside MCTS,
    where the world is determinized by construction (self-play: true hands;
    online: sampled hands), so feeding the opponents' cards is not cheating —
    it just stops throwing away information the search already has. The policy
    net (Big2Net) stays imperfect-information, since it must act in the real
    game and serve as the MCTS prior.

    No history GRU on purpose: history's role is inferring HIDDEN information,
    and there is none in a god-view evaluation — the static snapshot plus all
    four hands fully determines the strategic position. This also keeps the
    extra per-expansion forward pass cheap (pure MLP).

    Input:  static_feat (STATIC_DIM) + opp_hands (3×52, same layout as the
            belief target / features.encode_opp_hands)
    Output: (4,) tanh — expected normalized terminal reward per ABSOLUTE player.
    """

    def __init__(self, hidden_dim: int = 256, n_res_blocks: int = 4):
        super().__init__()
        input_dim = STATIC_DIM + OPP_HANDS_DIM
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res_blocks = nn.ModuleList(
            [ResBlock(hidden_dim) for _ in range(n_res_blocks)]
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Tanh(),
        )

    def forward(self, static_feat, opp_hands):
        x = torch.cat([static_feat, opp_hands], dim=-1)
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        return self.value_head(x)

    @torch.no_grad()
    def predict(self, static_feat, opp_hands):
        """Single-sample inference. Returns np.ndarray (4,)."""
        self.eval()
        sf = torch.FloatTensor(static_feat).unsqueeze(0)
        oh = torch.FloatTensor(opp_hands).unsqueeze(0)
        val = self.forward(sf, oh)
        return val.squeeze(0).cpu().numpy().astype(np.float32)
