"""V6 league RL — make the belief-aware policy actually USE its opponent read.

Warm-start from the V6 BC pretrain (PPO_V6), then self-play / league PPO. Two
losses per minibatch:
  - PPO (clip) + value : reward = engine Big2 score / reward_scale
  - belief BCE          : oracle = opponents' actual hands (FREE in self-play),
                          masked to the unseen pool. Keeps the belief head sharp.

Belief is detached into the policy (network_v6), so PPO gradients flow into
fuse/policy (learning to exploit the read) but NOT the belief head — the belief
head stays an accurate predictor, trained only by its BCE. Success check: after
RL, the belief-ablation action-flip% should rise well above the post-BC ~0%.

    ./.venv/bin/python -m ppo.train_v6_rl --updates 300 --tag v6rl
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from engine.env import Big2Env
from ppo.network_v6 import (CardAwareV6, E, build_batch_v6, belief_loss,
                            belief_target_from_env, unseen_mask_from_obs)
from ppo.network_cardaware import obs_from_env, legal_action_data, _to_batch
from ppo.ppo_trainer import compute_gae, lr_at
from ppo.eval_baselines import evaluate_vs_all

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DEF_INIT = os.path.join(CKPT_DIR, "saved", "PPO_V6.pt")


def _act(net, env, device, learner):
    """Sample a legal action. If learner, also return logp/value/rec (with belief oracle)."""
    legal_idx, feats = legal_action_data(env)
    obs = obs_from_env(env)
    ob = _to_batch([obs], device)
    af = torch.from_numpy(feats).unsqueeze(0).to(device)
    am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
    with torch.no_grad():
        score, value, _ = net(ob, af, am)
        logp_all = F.log_softmax(score, -1)
        a_idx = int(torch.multinomial(logp_all[0].exp(), 1).item())
    action = int(legal_idx[a_idx])
    if not learner:
        return action, None, None, None
    rec = {"obs": obs, "feats": feats, "pos": a_idx,
           "belief": belief_target_from_env(env), "bmask": unseen_mask_from_obs(obs)}
    return action, float(logp_all[0, a_idx]), float(value[0]), rec


def collect(env, net, pool, n_games, gamma, lam, reward_scale, learner_seats, device):
    records, logp_old, adv_all, ret_all = [], [], [], []
    for _ in range(n_games):
        env.reset()
        if pool:
            lseats = set(int(s) for s in np.random.choice([1, 2, 3, 4], learner_seats, replace=False))
            opp = {p: pool[np.random.randint(len(pool))] for p in range(1, 5) if p not in lseats}
        else:
            lseats, opp = {1, 2, 3, 4}, {}
        seat = {p: {"rec": [], "logp": [], "val": []} for p in lseats}
        rewards = None
        while not env.done:
            p = env.current_player
            if p in lseats:
                a, lp, v, rec = _act(net, env, device, True)
                seat[p]["rec"].append(rec); seat[p]["logp"].append(lp); seat[p]["val"].append(v)
            else:
                a = _act(opp[p], env, device, False)[0]
            rewards, _ = env.step(a)
        for sp in lseats:
            steps = len(seat[sp]["rec"])
            if steps == 0:
                continue
            r = np.zeros(steps, dtype=np.float32); r[-1] = float(rewards[sp - 1]) / reward_scale
            d = np.zeros(steps, dtype=bool); d[-1] = True
            adv, ret = compute_gae(r, seat[sp]["val"], d, gamma, lam)
            records.extend(seat[sp]["rec"]); logp_old.extend(seat[sp]["logp"])
            adv_all.extend(adv.tolist()); ret_all.extend(ret.tolist())
    return (records, np.asarray(logp_old, np.float32),
            np.asarray(adv_all, np.float32), np.asarray(ret_all, np.float32))


def update(net, opt, records, logp_old, adv, ret, device, epochs, minibatch,
           clip, vf, ent_coef, belief_coef, max_grad):
    n = len(records)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    lo = torch.from_numpy(logp_old).to(device)
    at = torch.from_numpy(adv).to(device); rt = torch.from_numpy(ret).to(device)
    idx = np.arange(n); acc = {"pi": 0., "v": 0., "ent": 0., "bel": 0., "kl": 0., "nb": 0}
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, n, minibatch):
            mb = idx[s:s + minibatch]
            b = build_batch_v6([records[i] for i in mb], device)
            score, value, bl = net(b["obs"], b["act_feats"], b["act_mask"])
            logp_all = F.log_softmax(score, -1)
            logp = logp_all.gather(1, b["pos"].unsqueeze(1)).squeeze(1)
            entropy = -(logp_all.exp() * logp_all).sum(-1)
            mbt = torch.from_numpy(mb).to(device)
            ratio = torch.exp(logp - lo[mbt]); a = at[mbt]
            pi_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
            v_loss = F.mse_loss(value, rt[mbt])
            ent = entropy.mean()
            bloss = belief_loss(bl, b["belief"], b["bmask"])
            loss = pi_loss + vf * v_loss - ent_coef * ent + belief_coef * bloss
            opt.zero_grad(); loss.backward(); clip_grad_norm_(net.parameters(), max_grad); opt.step()
            with torch.no_grad():
                acc["pi"] += pi_loss.item(); acc["v"] += v_loss.item(); acc["ent"] += ent.item()
                acc["bel"] += bloss.item(); acc["kl"] += (lo[mbt] - logp).mean().item(); acc["nb"] += 1
    nb = max(acc["nb"], 1)
    return {k: acc[k] / nb for k in ("pi", "v", "ent", "bel", "kl")}, n


@torch.no_grad()
def belief_pcount(net, device, n_games=20):
    """belief precision@count on the POLICY's own rollout states (the training
    distribution — random rollout would be OOD and read ~chance)."""
    env = Big2Env(); env.reset(); ps, pn = 0.0, 0
    for _ in range(n_games * 30):
        if env.done:
            env.reset()
        legal_idx, feats = legal_action_data(env)
        obs = obs_from_env(env)
        ob = _to_batch([obs], device)
        af = torch.from_numpy(feats).unsqueeze(0).to(device)
        am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
        score, _, bl = net(ob, af, am)
        P = torch.sigmoid(bl)[0].cpu().numpy() * unseen_mask_from_obs(obs)
        T = belief_target_from_env(env)
        for o in range(3):
            k = int(T[o].sum())
            if k:
                top = np.argpartition(-P[o], k - 1)[:k]
                ps += float(T[o, top].sum()) / k; pn += 1
        a_idx = int(torch.multinomial(F.log_softmax(score, -1)[0].exp(), 1).item())
        env.step(int(legal_idx[a_idx]))
    return ps / max(pn, 1)


def _load(path, device):
    net = CardAwareV6().to(device)
    net.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"])
    return net


def _snapshot(net, device):
    snap = CardAwareV6().to(device)
    snap.load_state_dict({k: v.detach().clone() for k, v in net.state_dict().items()})
    snap.eval()
    return snap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ckpt", default=DEF_INIT)
    ap.add_argument("--updates", type=int, default=300)
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
    ap.add_argument("--belief-coef", type=float, default=1.0)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--reward-scale", type=float, default=13.0)
    ap.add_argument("--learner-seats", type=int, default=2)
    ap.add_argument("--snapshot-every", type=int, default=50)
    ap.add_argument("--pool-max", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-games", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v6rl")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    warmup = max(1, int(args.warmup_frac * args.updates))

    env = Big2Env()
    net = _load(args.init_ckpt, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    pool = []  # starts empty -> pure self-play; snapshots grow the league
    print(f"V6 RL | warm-start {os.path.basename(args.init_ckpt)} | params {sum(p.numel() for p in net.parameters()):,} "
          f"| device {device} | belief_coef {args.belief_coef} | snapshot_every {args.snapshot_every}")
    print(f"   pre-RL belief P@count (self-play): {belief_pcount(net, device)*100:.1f}%")

    best = float("-inf")
    for upd in range(1, args.updates + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(upd, args.updates, args.lr, warmup)
        t0 = time.time()
        recs, lo, adv, ret = collect(env, net, pool, args.games_per_batch, args.gamma,
                                     args.lam, args.reward_scale, args.learner_seats, device)
        st, n = update(net, opt, recs, lo, adv, ret, device, args.epochs, args.minibatch,
                       args.clip, args.vf_coef, args.ent_coef, args.belief_coef, args.max_grad)
        print(f"upd {upd:3d}/{args.updates} | steps {n:5d} | lr {opt.param_groups[0]['lr']:.2e} "
              f"| pi {st['pi']:+.4f} | v {st['v']:.3f} | ent {st['ent']:.3f} | bel {st['bel']:.3f} "
              f"| kl {st['kl']:+.4f} | {time.time()-t0:.1f}s")

        if upd % args.eval_every == 0 or upd == args.updates:
            net.eval()
            res = evaluate_vs_all(lambda e: net.greedy(e, device), args.eval_games)
            bp = belief_pcount(net, device)
            net.train()
            print("   eval | " + " | ".join(
                f"{nm}: {wr*100:4.1f}% {sc:+.2f}{'*' if (wr > 0.25 and sc > 0) else ' '}"
                for nm, (wr, sc) in res.items()) + f" | belief P@cnt {bp*100:.1f}%")
            if res["smart"][1] > best:
                best = res["smart"][1]
                torch.save({"model": net.state_dict(), "arch": "v6", "update": upd,
                            "eval": res, "belief_prec": bp, "args": vars(args)},
                           os.path.join(CKPT_DIR, f"ppo_{args.tag}_best.pt"))
                print(f"   * new best (Smart {best:+.2f} @ upd {upd}) -> ppo_{args.tag}_best.pt")

        if args.snapshot_every > 0 and upd % args.snapshot_every == 0:
            pool.append(_snapshot(net, device))
            if len(pool) > args.pool_max:
                pool.pop(0)
            print(f"   + league snapshot @ upd {upd} (pool {len(pool)})")

        if upd % args.save_every == 0 or upd == args.updates:
            torch.save({"model": net.state_dict(), "arch": "v6", "update": upd, "args": vars(args)},
                       os.path.join(CKPT_DIR, f"ppo_{args.tag}_latest.pt"))


if __name__ == "__main__":
    main()
