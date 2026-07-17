"""V4PBV iteration: self-play (ISMCTS, rollout leaf) -> retrain policy+value (+human-BC mix)
-> two-tier local gate -> V4PBV_Iter{N} checkpoints. Repeat.

V4PBV = the currently-deployed system: PPO_V4 policy + BELIEF.pt (frozen) + VALUE_minplays.pt +
ISMCTS. This script replaces iterate_mode1.py (value-only) and iterate_mode1_policy.py
(policy-only) — both showed no measurable improvement. Root-caused this session: switching the
self-play LEAF from value_mp to rollout is what actually improves search decision quality (a
direct rollout-vs-policy diagnostic went from 33%->37% agreement, 25%->18% clear-miss); root
Dirichlet noise / min-visit floor ("de-anchoring") were tested at several strengths and showed no
further improvement, so this script does NOT enable them (eval_alpha1's _ROOT_NOISE/_MIN_VISITS
stay off — the flags remain available for future experiments).

Per iteration:
  1. self-play: all 4 seats play ISMCTS (frozen belief, rollout leaf, current policy as PUCT
     prior). Record (obs, legal, pi=visit-fraction) policy targets + the game schema (compatible
     with value_dataset.py / dominance_dataset.py).
  2. retrain POLICY: cross-entropy toward pi, mixed with a human-imitation batch (bc_dataset,
     target="human") so self-play doesn't drift from human-like play.
     retrain VALUE: ValueNet(extra_dim=1) on the growing self-play corpus + min_plays_to_empty,
     warm-started — kept in sync with the evolving policy for the (separate, fast) online leaf.
  3. gate: (a) margin vs a FROZEN V4 reference (sanity tripwire, known noisy) + (b) the
     search-vs-rollout agreement diagnostic (more sensitive signal from this session).
  4. save V4PBV_Iter{N}_policy.pt / V4PBV_Iter{N}_value.pt. Belief never retrained.

Local gates are sanity checks, not the judge — see kind-napping-wolf.md: the real test is
periodic online play (swap checkpoints into cardaware_wrapper.py, accumulate avg_score).

    python iterate_v4pbv.py --iters 3 --games-per-iter 80 --sims 60
"""
import bootstrap  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
import torch

import eval_alpha1 as E
from ppo.network_cardaware import CardAwareActorCritic
from ppo.action_features import ACTION_FEATURES, AF
from ppo.bc_dataset import build_records as build_bc_records
from value_model import ValueNet, build_value_batch
from value_dataset import build_value_records
from hand_features import min_plays_to_empty

DEV = E.DEV
CKPT_DIR = bootstrap.CKPT_DIR
CORPUS = os.path.join(bootstrap.DATA_DIR, "v4pbv_selfplay.jsonl")

# Self-play search config (Stage 1 finding: rollout leaf is the real win; de-anchoring is not).
E._LEAF = "rollout"
E._USE_BELIEF = True
E._ROOT_NOISE = False
E._MIN_VISITS = 0

# FROZEN V4 reference for the sanity gate (E._policy evolves each iteration, so it can't be the
# yardstick — margin_vs_v4 needs a fixed opponent).
_V4 = CardAwareActorCritic().to(DEV)
_V4.load_state_dict(torch.load(f"{E.SAVED}/PPO_V4.pt", map_location=DEV)["model"]); _V4.eval()


@torch.no_grad()
def _v4_greedy(env):
    legal, feats = E.legal_action_data(env)
    ob = E._to_batch([E.obs_from_env(env)], DEV)
    af = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    am = torch.ones(1, len(legal), dtype=torch.bool, device=DEV)
    logits, _ = _V4.forward(ob, af, am)
    return int(legal[int(logits[0].argmax())])


def play_and_record(sims, cpuct):
    """One all-4-ISMCTS self-play game -> (policy targets, game-schema dict)."""
    env = E.Big2Env(); env.reset()
    hands = {str(s): [int(c) for c in env.game.currentHands[s + 1]] for s in range(4)}
    pol_targets, plays, steps = [], [], 0
    rewards = env.game.rewards
    while not env.done and steps < 300:
        p = env.current_player
        before = set(int(c) for c in env.game.currentHands[p])
        obs = E.obs_from_env(env)
        legal, _ = E.legal_action_data(env)
        best, visits = E.alpha1_action(env, p, sims, cpuct, return_visits=True)
        tot = sum(visits.values()) if visits else 0
        if len(legal) > 1 and tot > 0:
            pi = np.array([visits.get(int(a), 0) / tot for a in legal], dtype=np.float32)
            pol_targets.append({"obs": obs, "feats": ACTION_FEATURES[legal].astype(np.float32), "pi": pi})
        rewards, _ = env.step(best)
        after = set(int(c) for c in env.game.currentHands[p])
        plays.append({"seat": p - 1, "action": "pass" if best == E.PASS_IDX else "play",
                      "cards": sorted(before - after)})
        steps += 1
    scores = {str(s): int(rewards[s]) for s in range(4)}
    winner = next((s for s in range(4) if env.game.currentHands[s + 1].size == 0), 0)
    game_rec = {"hands": hands, "plays": plays, "scores": scores, "winner": winner, "source": "v4pbv"}
    return pol_targets, game_rec


def selfplay(n_games, sims, cpuct):
    t0 = time.time(); pol_buf = []
    with open(CORPUS, "a") as f:
        for i in range(n_games):
            pt, gr = play_and_record(sims, cpuct)
            pol_buf += pt
            f.write(json.dumps(gr) + "\n")
            if (i + 1) % 10 == 0:
                f.flush()
                print(f"    self-play {i+1}/{n_games} ({(time.time()-t0)/(i+1):.1f}s/game, "
                      f"{len(pol_buf)} pi-targets)")
    total = sum(1 for _ in open(CORPUS))
    print(f"    self-play done: +{n_games} -> {total} games total in {os.path.basename(CORPUS)}")
    return pol_buf


def _bc_as_common(n):
    """n human-imitation targets (one-hot pi over that decision's legal set) in the same record
    shape as self-play pi-targets, so both can share one batch builder."""
    if n <= 0:
        return []
    recs = build_bc_records(target="human", verbose=False)
    idx = np.random.choice(len(recs), size=min(n, len(recs)), replace=False)
    out = []
    for i in idx:
        r = recs[i]; L = r["feats"].shape[0]
        pi = np.zeros(L, dtype=np.float32); pi[r["pos"]] = 1.0
        out.append({"obs": r["obs"], "feats": r["feats"], "pi": pi})
    return out


def build_policy_batch(records):
    obs = E._to_batch([r["obs"] for r in records], DEV)
    Lmax = max(r["feats"].shape[0] for r in records); B = len(records)
    feats = np.zeros((B, Lmax, AF), dtype=np.float32)
    mask = np.zeros((B, Lmax), dtype=bool)
    pi = np.zeros((B, Lmax), dtype=np.float32)
    for i, r in enumerate(records):
        L = r["feats"].shape[0]; feats[i, :L] = r["feats"]; mask[i, :L] = True; pi[i, :L] = r["pi"]
    return (obs, torch.from_numpy(feats).to(DEV), torch.from_numpy(mask).to(DEV),
            torch.from_numpy(pi).to(DEV))


def train_policy(sp_targets, warm, epochs, bc_mix_ratio, lr=5e-4, wd=1e-4, batch=256):
    bc_n = int(len(sp_targets) * bc_mix_ratio / max(1 - bc_mix_ratio, 1e-6))
    buffer = sp_targets + _bc_as_common(bc_n)
    print(f"    train_policy: {len(sp_targets)} self-play + {bc_n} human-BC = {len(buffer)} targets")
    net = CardAwareActorCritic().to(DEV); net.load_state_dict(warm.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    idx = np.arange(len(buffer))
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), batch):
            obs, feats, mask, pi = build_policy_batch([buffer[i] for i in idx[s:s + batch]])
            logits, _ = net.forward(obs, feats, mask)
            loss = -(pi * torch.log_softmax(logits, -1)).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net


def train_value(warm_net, epochs, lr=1e-3, wd=3e-4, batch=256):
    recs = build_value_records(path=CORPUS, samples_per_game=1, verbose=False)
    for r in recs:
        hand = [int(c) for c in r["obs"]["hand_ids"] if c > 0]
        r["feat"] = np.array([min_plays_to_empty(hand) / 13.0], dtype=np.float32)
    idx = np.arange(len(recs)); np.random.shuffle(idx)
    nval = max(1, int(len(recs) * 0.1)); val_i, tr_i = idx[:nval], idx[nval:]
    net = ValueNet(extra_dim=1).to(DEV)
    if warm_net is not None:
        net.load_state_dict(warm_net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        np.random.shuffle(tr_i)
        for s in range(0, len(tr_i), batch):
            b = build_value_batch([recs[i] for i in tr_i[s:s + batch]], DEV)
            loss = torch.nn.functional.mse_loss(net(b["obs"]), b["target"])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        P, Y = [], []
        for s in range(0, len(val_i), 512):
            b = build_value_batch([recs[i] for i in val_i[s:s + 512]], DEV)
            P += net(b["obs"]).cpu().numpy().tolist(); Y += b["target"].cpu().numpy().tolist()
    P, Y = np.array(P), np.array(Y)
    mse = float(np.mean((P - Y) ** 2)); base = float(np.mean((Y.mean() - Y) ** 2))
    return net, mse, base, len(recs)


def margin_vs_v4(sims, cpuct, games):
    """ISMCTS(current E._policy) at one seat vs FROZEN V4 at the other three."""
    a = []
    for i in range(games):
        seat = (i % 4) + 1
        env = E.Big2Env(); env.reset(); steps = 0
        while not env.done and steps < 300:
            p = env.current_player
            act = E.alpha1_action(env, p, sims, cpuct) if p == seat else _v4_greedy(env)
            env.step(act); steps += 1
        a.append(float(env.game.rewards[seat - 1]))
    return float(np.mean(a)) * 4.0 / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--games-per-iter", type=int, default=80)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--policy-epochs", type=int, default=4)
    ap.add_argument("--value-epochs", type=int, default=12)
    ap.add_argument("--bc-mix", type=float, default=0.25, help="fraction of policy training batch that is human-imitation")
    ap.add_argument("--buffer-iters", type=int, default=3, help="rolling window of self-play iterations kept for policy targets")
    ap.add_argument("--gate-games", type=int, default=40)
    ap.add_argument("--gate-decisions", type=int, default=20, help="measure_disagreement sample size per gate")
    ap.add_argument("--gate-k-worlds", type=int, default=8)
    ap.add_argument("--start-iter", type=int, default=0, help="V4PBV_Iter{N} label to start saving from")
    ap.add_argument("--resume-policy", default="", help="policy ckpt to warm-start from (default: PPO_V4.pt baseline)")
    ap.add_argument("--resume-value", default="", help="value ckpt to warm-start from (default: VALUE_minplays.pt baseline)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed); torch.manual_seed(args.seed)

    E._load_aug()  # populates E._value_mp (baseline VALUE_minplays.pt) + E._dom (unused here)
    if args.resume_policy:
        E._policy.load_state_dict(torch.load(args.resume_policy, map_location=DEV)["model"])
        print(f"resumed policy <- {os.path.basename(args.resume_policy)}")
    if args.resume_value:
        E._value_mp.load_state_dict(torch.load(args.resume_value, map_location=DEV)["model"])
        print(f"resumed value  <- {os.path.basename(args.resume_value)}")

    print(f"V4PBV iteration | leaf={E._LEAF} root_noise={E._ROOT_NOISE} min_visits={E._MIN_VISITS} "
          f"belief=frozen | sims={args.sims} cpuct={args.cpuct} bc_mix={args.bc_mix} | "
          f"{args.iters} iters x {args.games_per_iter} games")

    pol_buffer = []   # rolling window is per-invocation; resuming starts this empty (refills over --buffer-iters iters)
    for k in range(args.iters):
        it = args.start_iter + k
        print(f"\n=== V4PBV_Iter{it} ===")
        pol_buffer += selfplay(args.games_per_iter, args.sims, args.cpuct)
        cap = args.buffer_iters * args.games_per_iter * 55       # ~55 pi-targets/game
        pol_buffer = pol_buffer[-cap:]

        new_policy = train_policy(pol_buffer, E._policy, args.policy_epochs, args.bc_mix)
        new_value, vmse, vbase, vn = train_value(E._value_mp, args.value_epochs)
        E._policy = new_policy
        E._value_mp = new_value

        torch.save({"model": new_policy.state_dict(), "arch": "cardaware", "iter": it},
                   os.path.join(CKPT_DIR, f"V4PBV_Iter{it}_policy.pt"))
        torch.save({"model": new_value.state_dict(), "arch": "value", "extra_dim": 1, "iter": it,
                   "val_mse": vmse}, os.path.join(CKPT_DIR, f"V4PBV_Iter{it}_value.pt"))

        m = margin_vs_v4(args.sims, args.cpuct, args.gate_games)
        dg = E.measure_disagreement(n_decisions=args.gate_decisions, k_worlds=args.gate_k_worlds,
                                     use_search=True, sims=args.sims, cpuct=args.cpuct, verbose=False)
        print(f"  V4PBV_Iter{it}: value val_mse {vmse:.4f} (base {vbase:.4f}, {vn} recs) | "
              f"vs-V4 margin {m:+.2f}/game ({args.gate_games} games) | "
              f"search-vs-rollout agree {100*dg['agree_rate']:.0f}% clear-miss {100*dg['clear_miss_rate']:.0f}% "
              f"({dg['n']} decisions) | pi-buffer {len(pol_buffer)} targets")


if __name__ == "__main__":
    main()
