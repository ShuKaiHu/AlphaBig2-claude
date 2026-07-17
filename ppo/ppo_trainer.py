"""Self-play PPO trainer for Big 2 — paper-faithful, architecture-agnostic.

Follows Patwa, *Self-Play RL under Imperfect Information in Big 2*
(arXiv 2605.28863):
  * current-policy self-play (all 4 seats = current policy; train on every seat)
  * PPO: 4 epochs, clip 0.2, gamma 0.99, lambda 0.95, value coef 0.5,
    entropy coef 0.05, global grad-norm clip 0.5, minibatch 256,
    64 full games per batch, lr 3e-5 with warmup + cosine decay
  * eval vs fixed Random / Greedy / Smart pools (success = win>25% & score>0)

--arch mlp        : 412-bit state + 14739 masked head (paper's old baseline)
--arch cardaware  : card embedding + self-attention + dot-product (paper)

Deviations (documented): reward is our engine's *multiplied* Big 2 score (matches
real online scoring — the actual objective); the learning signal is divided by
--reward-scale for stability while eval reports the TRUE score. Engine has no
plain flush (see eval_baselines.py).

Run from the worktree root:
    ./.venv/bin/python -m ppo.ppo_trainer --arch cardaware --updates 200
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from engine.env import Big2Env
from ppo.policies import make_policy
from ppo.eval_baselines import evaluate_vs_all
from ppo.pool_eval import evaluate_vs_pool, evaluate_vs_each, load_pool

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def compute_gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def collect(env, policy, n_games, gamma, lam, reward_scale, pool=None, learner_seats=4):
    """Play n_games and gather the LEARNER's transitions.

    pool empty/None  -> pure current-policy self-play (all 4 seats = learner).
    pool non-empty   -> league: `learner_seats` random seats use the current
      policy (collected & trained on); the other seats are each controlled by a
      frozen opponent sampled from `pool` (sampled actions, NOT collected)."""
    records, logp_old, adv_all, ret_all = [], [], [], []

    for _ in range(n_games):
        env.reset()
        if pool:
            lseats = set(int(s) for s in np.random.choice(
                [1, 2, 3, 4], size=learner_seats, replace=False))
            opp = {p: pool[np.random.randint(len(pool))]
                   for p in range(1, 5) if p not in lseats}
        else:
            lseats, opp = {1, 2, 3, 4}, {}
        seat = {p: {"rec": [], "logp": [], "val": []} for p in lseats}
        rewards = None
        while not env.done:
            p = env.current_player
            if p in lseats:
                action, logp, value, rec = policy.act(env)
                seat[p]["rec"].append(rec)
                seat[p]["logp"].append(logp)
                seat[p]["val"].append(value)
            else:
                action = opp[p].act(env)[0]   # frozen opponent (sampled), not collected
            rewards, _done = env.step(action)

        for sp in lseats:
            steps = len(seat[sp]["rec"])
            if steps == 0:
                continue
            r = np.zeros(steps, dtype=np.float32)
            r[-1] = float(rewards[sp - 1]) / reward_scale
            d = np.zeros(steps, dtype=bool)
            d[-1] = True
            adv, ret = compute_gae(r, seat[sp]["val"], d, gamma, lam)
            records.extend(seat[sp]["rec"])
            logp_old.extend(seat[sp]["logp"])
            adv_all.extend(adv.tolist())
            ret_all.extend(ret.tolist())

    return (records,
            np.asarray(logp_old, dtype=np.float32),
            np.asarray(adv_all, dtype=np.float32),
            np.asarray(ret_all, dtype=np.float32))


def _load_frozen(path, arch, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    p = make_policy(ck.get("arch", arch), device)
    p.net.load_state_dict(ck["model"])
    p.eval()
    return p


def _snapshot(policy, arch, device):
    """A frozen deep copy of the current policy (a new league member)."""
    snap = make_policy(arch, device)
    snap.net.load_state_dict({k: v.detach().clone()
                              for k, v in policy.net.state_dict().items()})
    snap.eval()
    return snap


def ppo_update(policy, opt, records, logp_old, adv, ret, device,
               epochs, minibatch, clip, vf_coef, ent_coef, max_grad):
    n = len(records)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    logp_old_t = torch.from_numpy(logp_old).to(device)
    adv_t = torch.from_numpy(adv).to(device)
    ret_t = torch.from_numpy(ret).to(device)
    idx = np.arange(n)
    acc = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "nb": 0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            mb = idx[s:s + minibatch]
            batch = policy.build_minibatch([records[i] for i in mb], device)
            logp, entropy, value = policy.evaluate(batch)
            mb_t = torch.from_numpy(mb).to(device)
            ratio = torch.exp(logp - logp_old_t[mb_t])
            a = adv_t[mb_t]
            pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            v_loss = nn.functional.mse_loss(value, ret_t[mb_t])
            ent = entropy.mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            opt.step()
            with torch.no_grad():
                acc["pi"] += pi_loss.item()
                acc["v"] += v_loss.item()
                acc["ent"] += ent.item()
                acc["kl"] += (logp_old_t[mb_t] - logp).mean().item()
                acc["nb"] += 1
    nb = max(acc["nb"], 1)
    return {k: acc[k] / nb for k in ("pi", "v", "ent", "kl")}, n


def lr_at(update, total, base_lr, warmup):
    if update <= warmup:
        return base_lr * update / max(1, warmup)
    prog = (update - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))


def run_eval(policy, n_games):
    policy.eval()
    res = evaluate_vs_all(policy.greedy, n_games)
    policy.train()
    return res


def run_eval_pool(policy, eval_pool, n_games, device, seed):
    """Checkpoint-selection criterion validated online 2026-07-06: scoring vs a
    diverse historical opponent pool beats vs a single fixed opponent (Smart)
    and beats picking by training loss alone -- see the
    policy-checkpoint-selection-vs-pool memory. Used for BOTH the
    early-stopping/best-checkpoint decision AND periodic progress reporting;
    run_eval/evaluate_vs_all above is kept only as an informational side-metric."""
    policy.eval()
    wr, sc = evaluate_vs_pool(policy.greedy, eval_pool, n_games, device=device, seed=seed)
    policy.train()
    return wr, sc


def run_eval_per_member(policy, eval_pool, n_games, device):
    """Score the candidate against EACH eval_pool member individually (that
    member alone occupies all 3 opponent seats) -- for gating a checkpoint on
    "must beat every member", not just the aggregate mixed-table score, which
    a checkpoint can pass by being fine on average while still losing badly to
    one specific member (confirmed this session: a strong existing model beat
    one archetype +9.75 while losing to two others, invisible in the
    aggregate). Returns ({name: (win_rate, avg_score)}, min_score)."""
    policy.eval()
    res = evaluate_vs_each(policy.greedy, eval_pool, n_games, device=device)
    policy.train()
    min_score = min(sc for _, sc in res.values())
    return res, min_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["mlp", "cardaware", "cardaware_history"], default="cardaware")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--games-per-batch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--ent-coef", type=float, default=0.05)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=13.0)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="cardaware", help="checkpoint filename tag")
    # ── league / opponent-pool self-play ──
    ap.add_argument("--init-ckpt", type=str, default="", help="warm-start weights from this checkpoint")
    ap.add_argument("--pool-ckpts", type=str, default="", help="comma-separated frozen opponent checkpoints (enables league)")
    ap.add_argument("--learner-seats", type=int, default=2, help="seats the learner controls per game (rest = pool)")
    ap.add_argument("--snapshot-every", type=int, default=0, help="add a frozen snapshot of the current policy to the pool every N updates (0=off)")
    ap.add_argument("--pool-max", type=int, default=20, help="max pool size (drop oldest)")
    # ── restricted eval pool + per-member gating ("must beat each individually") ──
    ap.add_argument("--eval-pool-ckpts", type=str, default="",
                     help="comma-separated checkpoints to use as the EVAL/gating pool instead of the "
                          "full historical pool (e.g. the 5 archetype models) -- independent of "
                          "--pool-ckpts, which only affects self-play data collection")
    ap.add_argument("--gate-per-member", action="store_true",
                     help="checkpoint selection uses min(score vs each eval-pool member individually) "
                          "instead of the aggregate mixed-table pool score -- a checkpoint only counts "
                          "as 'best' if it improves the WORST individual matchup, not just the average")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    os.makedirs(CKPT_DIR, exist_ok=True)
    warmup = max(1, int(args.warmup_frac * args.updates))

    env = Big2Env()
    policy = make_policy(args.arch, device)
    if args.init_ckpt:
        _ck = torch.load(args.init_ckpt, map_location=device, weights_only=False)
        policy.net.load_state_dict(_ck["model"])
        print(f"warm-start from {os.path.basename(args.init_ckpt)} (upd {_ck.get('update')})")
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    nparams = sum(p.numel() for p in policy.parameters())

    pool = [_load_frozen(p.strip(), args.arch, device)
            for p in args.pool_ckpts.split(",") if p.strip()]
    league = bool(pool) or args.snapshot_every > 0
    print(f"device={device}  arch={args.arch}  params={nparams:,}  warmup={warmup}  "
          f"league={league} pool={len(pool)} learner_seats={args.learner_seats if league else 4}")

    if args.eval_pool_ckpts:
        eval_members = [(os.path.splitext(os.path.basename(p.strip()))[0], p.strip(), args.arch)
                         for p in args.eval_pool_ckpts.split(",") if p.strip()]
        eval_pool = load_pool(device, members=eval_members)
    else:
        eval_pool = load_pool(device)
    if args.gate_per_member and not eval_pool:
        raise ValueError("--gate-per-member requires a non-empty eval pool")
    best_pool_score = float("-inf")
    best_min_score = float("-inf")
    for update in range(1, args.updates + 1):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(update, args.updates, args.lr, warmup)
        t0 = time.time()
        records, logp_old, adv, ret = collect(
            env, policy, args.games_per_batch, args.gamma, args.lam, args.reward_scale,
            pool=(pool if league else None), learner_seats=args.learner_seats)
        st, n_steps = ppo_update(policy, opt, records, logp_old, adv, ret, device,
                                 args.epochs, args.minibatch, args.clip,
                                 args.vf_coef, args.ent_coef, args.max_grad)
        dt = time.time() - t0
        print(f"upd {update:3d}/{args.updates} | steps {n_steps:5d} | lr {opt.param_groups[0]['lr']:.2e} "
              f"| pi {st['pi']:+.4f} | v {st['v']:.3f} | ent {st['ent']:.3f} | kl {st['kl']:+.4f} | {dt:.1f}s")

        if update % args.eval_every == 0 or update == args.updates:
            res = run_eval(policy, args.eval_games)
            line = "   eval | " + " | ".join(
                f"{n}: {wr*100:4.1f}% {sc:+.2f}{'*' if (wr > 0.25 and sc > 0) else ' '}"
                for n, (wr, sc) in res.items())
            print(line + "   (* = win>25% & score>0, informational only)")
            # checkpoint selection uses vs-opponent-POOL score, not vs-Smart --
            # validated online 2026-07-06 (see policy-checkpoint-selection-vs-pool
            # memory): vs-pool beats vs-single-opponent and beats picking by loss.
            pool_wr, pool_sc = run_eval_pool(policy, eval_pool, args.eval_games, device, seed=update)
            print(f"   pool eval | win_rate {pool_wr*100:4.1f}% | avg_score {pool_sc:+.2f}")

            if args.gate_per_member:
                per_member, min_score = run_eval_per_member(
                    policy, eval_pool, max(1, args.eval_games // len(eval_pool)), device)
                member_line = "   per-member | " + " | ".join(
                    f"{n}: {wr*100:4.1f}% {sc:+.2f}" for n, (wr, sc) in per_member.items())
                print(member_line + f"   | min={min_score:+.2f}")
                if min_score > best_min_score:
                    best_min_score = min_score
                    torch.save({"model": policy.state_dict(), "update": update,
                                "arch": args.arch, "pool_score": pool_sc, "pool_win_rate": pool_wr,
                                "min_member_score": min_score, "per_member": per_member,
                                "eval": res, "args": vars(args)},
                               os.path.join(CKPT_DIR, f"ppo_{args.tag}_best.pt"))
                    print(f"   * new best (min-per-member score {min_score:+.2f} @ upd {update}) "
                          f"-> ppo_{args.tag}_best.pt")
            elif pool_sc > best_pool_score:
                best_pool_score = pool_sc
                torch.save({"model": policy.state_dict(), "update": update,
                            "arch": args.arch, "pool_score": pool_sc, "pool_win_rate": pool_wr,
                            "eval": res, "args": vars(args)},
                           os.path.join(CKPT_DIR, f"ppo_{args.tag}_best.pt"))
                print(f"   * new best (pool avg_score {pool_sc:+.2f} @ upd {update}) -> ppo_{args.tag}_best.pt")

        # league growth: periodically freeze the current policy into the pool
        if args.snapshot_every > 0 and update % args.snapshot_every == 0:
            pool.append(_snapshot(policy, args.arch, device))
            if len(pool) > args.pool_max:
                pool.pop(0)
            league = True
            print(f"   + league snapshot @ upd {update} (pool size {len(pool)})")

        if update % args.save_every == 0 or update == args.updates:
            payload = {"model": policy.state_dict(), "update": update,
                       "arch": args.arch, "args": vars(args)}
            torch.save(payload, os.path.join(CKPT_DIR, f"ppo_{args.tag}_latest.pt"))


if __name__ == "__main__":
    main()
