"""Actor-critic network for Big2 PPO.

Input  : 412-dim state vector from big2Game.getCurrentState()
Policy : logits over ACTION_SIZE (14739) actions, masked to legal moves
Value  : scalar state-value estimate

Independent of the engine's AlphaZero net (engine/model.py) on purpose —
this is the new PPO experiment's own model.
"""
import torch
import torch.nn as nn

OBS_DIM = 412


class ActorCritic(nn.Module):
    def __init__(self, action_size: int, hidden: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(OBS_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, action_size)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs, action_mask):
        """obs: (B, 412) float; action_mask: (B, A) additive (0 legal / -inf illegal)."""
        h = self.body(obs)
        # clamp the additive mask so illegal actions get a large-but-finite penalty
        mask = torch.clamp(action_mask, min=-1e9)
        logits = self.policy_head(h) + mask
        value = self.value_head(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, obs, action_mask):
        """Sample one action. Returns (action, logprob, value)."""
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs, action_mask, actions):
        """For the PPO update: logprob of taken actions, entropy, value."""
        logits, value = self.forward(obs, action_mask)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
