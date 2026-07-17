"""Uniform policy interface so the PPO trainer is architecture-agnostic.

Both policies expose:
    act(env)            -> (action, logp, value, record)   # sample, for collection
    greedy(env)         -> action                           # argmax, for eval
    build_minibatch(records, device) -> batch
    evaluate(batch)     -> (logp, entropy, value)           # for the PPO update
    parameters(), train(), eval(), state_dict()

  * MLPPolicy       — 412-bit state + 14739 masked head (Phase A)
  * CardAwarePolicy — card embedding + self-attention + dot-product (Phase B)
"""
import numpy as np
import torch

from engine.env import Big2Env
from ppo.network import ActorCritic
from ppo import network_cardaware as CA
from ppo import network_cardaware_history as CAH
from ppo import network_v6 as V6


class MLPPolicy:
    arch = "mlp"

    def __init__(self, device):
        self.device = device
        self.net = ActorCritic(Big2Env.ACTION_SIZE).to(device)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    def _obs(self, env):
        _p, state, avail = env.game.getCurrentState()
        return state.reshape(-1).astype(np.float32), avail.reshape(-1).astype(np.float32)

    @torch.no_grad()
    def act(self, env):
        obs, mask = self._obs(env)
        ot = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        action, logp, value = self.net.act(ot, mt)
        a = int(action.item())
        return a, float(logp.item()), float(value.item()), {"obs": obs, "mask": mask, "act": a}

    @torch.no_grad()
    def greedy(self, env):
        obs, mask = self._obs(env)
        ot = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        mt = torch.from_numpy(mask).unsqueeze(0).to(self.device)
        logits, _ = self.net(ot, mt)
        return int(torch.argmax(logits, dim=-1).item())

    def build_minibatch(self, records, device):
        return {
            "obs": torch.tensor(np.array([r["obs"] for r in records]), dtype=torch.float32, device=device),
            "mask": torch.tensor(np.array([r["mask"] for r in records]), dtype=torch.float32, device=device),
            "act": torch.tensor([r["act"] for r in records], dtype=torch.long, device=device),
        }

    def evaluate(self, batch):
        return self.net.evaluate(batch["obs"], batch["mask"], batch["act"])


class CardAwarePolicy:
    arch = "cardaware"

    def __init__(self, device):
        self.device = device
        self.net = CA.CardAwareActorCritic().to(device)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    @torch.no_grad()
    def act(self, env):
        return self.net.act(env, self.device)

    @torch.no_grad()
    def greedy(self, env):
        return self.net.greedy(env, self.device)

    def build_minibatch(self, records, device):
        return CA.build_batch(records, device)

    def evaluate(self, batch):
        return CA.evaluate_batch(self.net, batch)


class CardAwareHistoryPolicy:
    arch = "cardaware_history"

    def __init__(self, device):
        self.device = device
        self.net = CAH.CardAwareActorCriticHistory().to(device)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    @torch.no_grad()
    def act(self, env):
        return self.net.act(env, self.device)

    @torch.no_grad()
    def greedy(self, env):
        return self.net.greedy(env, self.device)

    def build_minibatch(self, records, device):
        return CAH.build_batch_history(records, device)

    def evaluate(self, batch):
        return CAH.evaluate_batch(self.net, batch)


class CardAwareV6Policy:
    arch = "v6"

    def __init__(self, device):
        self.device = device
        self.net = V6.CardAwareV6().to(device)

    def parameters(self):
        return self.net.parameters()

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def state_dict(self):
        return self.net.state_dict()

    @torch.no_grad()
    def act(self, env):
        return self.net.act(env, self.device)

    @torch.no_grad()
    def greedy(self, env):
        return self.net.greedy(env, self.device)

    def build_minibatch(self, records, device):
        return CA.build_batch(records, device)

    def evaluate(self, batch):
        return V6.evaluate_batch_v6(self.net, batch)


def make_policy(arch, device):
    if arch == "mlp":
        return MLPPolicy(device)
    if arch == "cardaware":
        return CardAwarePolicy(device)
    if arch == "cardaware_history":
        return CardAwareHistoryPolicy(device)
    if arch == "v6":
        return CardAwareV6Policy(device)
    raise ValueError(f"unknown arch: {arch}")
