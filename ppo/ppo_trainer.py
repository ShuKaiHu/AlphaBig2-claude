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
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from engine.env import Big2Env
from ppo.policies import make_policy
from ppo.eval_baselines import evaluate_vs_all
from ppo.pool_eval import evaluate_vs_pool, evaluate_vs_each, load_pool

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# first few luck-baseline subtractions are printed for eyeball verification
_LUCK_DEBUG = {"left": 0}
# first few injected episode starts are printed for eyeball verification
_INJ_DEBUG = {"left": 0}


def _start_d2_episode(env, rec):
    """Start an episode from a D2 mid-game pool record (M3 Phase 1, Track A).

    SEAT CONVENTION: pool records store 0-BASED seats (0-3);
    big2Game.restore_state wants 1-BASED engine players (1-4). The +1
    conversion happens HERE and only here. The `hands` dict from
    d2_state_args is keyed 0-3, which restore_state's _normalize_hands_arg
    maps to players seat+1 itself — consistent with the explicit +1 on
    whose_turn / trick_owner / passed_players below, and with the
    cards_played (4,52) mask whose row s is player s+1.

    Returns the 1-BASED learner target seat = the record's caged-loss seat."""
    from ppo.injection_pools import d2_state_args

    from big2Game import big2Game
    acting, remaining, cards_played, control, trick_cards, trick_owner, passed = \
        d2_state_args(rec)
    g = big2Game.restore_state(
        hands=remaining,                        # {0..3}-keyed, mapped to seat+1
        whose_turn=acting + 1,
        trick_cards=(None if control else trick_cards),
        trick_owner=(None if control else trick_owner + 1),
        passed_players=tuple(q + 1 for q in passed),
        cards_played=cards_played,              # (4,52) mask, row s = player s+1
        must_play_club3=False)                  # mid-game: never force 3C
    env.load_game(g)
    return int(rec["seat"]) + 1


def _start_d1_episode(env, rec):
    """Start an episode from a D1 strong-full-deal pool record (M3 Phase 1).

    A D1 record is a FULL 13-card deal, so this is a normal reset(hands=...):
    the 3C holder starts and the forced-3C opening applies, exactly like a
    natural deal — only the cards are a real online deal instead of random.
    Pool seats are 0-based; reset(hands=) accepts the 0-3-keyed dict directly.

    Returns the 1-BASED learner target seat = strong_seat + 1."""
    env.reset(hands={s: rec["hands"][str(s)] for s in range(4)})
    return int(rec["strong_seat"]) + 1


def compute_gae(rewards, values, dones, gamma, lam):
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = 0.0 if dones[t] else values[t + 1]
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (0.0 if dones[t] else last)
        adv[t] = last
    return adv, adv + np.asarray(values, dtype=np.float32)


def collect(env, policy, n_games, gamma, lam, reward_scale, pool=None, learner_seats=4,
            luck_fn=None, inject_d2=0.0, inject_d1=0.0, d2_pool=None, d1_pool=None):
    """Play n_games and gather the LEARNER's transitions.

    pool empty/None  -> pure current-policy self-play (all 4 seats = learner).
    pool non-empty   -> league: `learner_seats` random seats use the current
      policy (collected & trained on); the other seats are each controlled by a
      frozen opponent sampled from `pool` (sampled actions, NOT collected).

    luck_fn (default None = exact legacy behavior): maps the learner seat's
      DEALT 13-card hand -> E[score | dealt hand]; subtracted from the terminal
      reward (deal-luck baseline, M3 Track C). Fit on 13-card STARTING hands
      only — for injected D2 MID-GAME episode starts it is hard-skipped
      (dealt hand recorded as None -> luck_fn returns 0.0; NOT a len!=13
      check, because a mid-game seat can still hold exactly 13 cards).
      Injected D1 episodes are real 13-card deals, so the baseline applies.

    inject_d2 / inject_d1 (M3 Phase 1, defaults 0.0 = exact legacy behavior,
      including the RNG stream — the per-episode uniform draw only happens
      when a fraction is > 0): fractions of episodes started from D2 mid-game
      caged-loss states (big2Game.restore_state) / D1 strong full deals
      (reset(hands=...)) sampled uniformly from d2_pool / d1_pool (TRAIN
      split). LEARNER SEAT for injected episodes: the trainer picks learner
      seats per-episode (not per-worker), so we simply FORCE the pool
      record's target seat into lseats (plus learner_seats-1 random others)
      — the pool state's seats are never remapped, preserving the state's
      semantics (turn order, trick owner, passed set) exactly as harvested.

    Returns (records, logp_old, adv, ret, mix) where mix counts episode
      starts: {"natural": n, "d2": n, "d1": n}."""
    records, logp_old, adv_all, ret_all = [], [], [], []
    mix = {"natural": 0, "d2": 0, "d1": 0}

    for _ in range(n_games):
        target_seat, kind = None, "natural"
        if inject_d2 > 0.0 or inject_d1 > 0.0:
            u = float(np.random.rand())
            if d2_pool and u < inject_d2:
                rec = d2_pool[np.random.randint(len(d2_pool))]
                target_seat = _start_d2_episode(env, rec)
                kind = "d2"
            elif d1_pool and u < inject_d2 + inject_d1:
                rec = d1_pool[np.random.randint(len(d1_pool))]
                target_seat = _start_d1_episode(env, rec)
                kind = "d1"
            else:
                env.reset()
            if kind != "natural" and _INJ_DEBUG["left"] > 0:
                _INJ_DEBUG["left"] -= 1
                extra = (" | luck baseline hard-skipped (mid-game)"
                         if (kind == "d2" and luck_fn is not None) else "")
                print(f"   [inject] {kind} game_id={rec['game_id']} "
                      f"event_idx={rec.get('event_idx', '-')} "
                      f"learner_seat={target_seat} (pool seat "
                      f"{target_seat - 1} 0-based){extra}")
        else:
            env.reset()
        mix[kind] += 1
        if pool:
            if target_seat is not None:
                others = [p for p in (1, 2, 3, 4) if p != target_seat]
                lseats = {target_seat} | set(
                    int(s) for s in np.random.choice(
                        others, size=learner_seats - 1, replace=False))
            else:
                lseats = set(int(s) for s in np.random.choice(
                    [1, 2, 3, 4], size=learner_seats, replace=False))
            opp = {p: pool[np.random.randint(len(pool))]
                   for p in range(1, 5) if p not in lseats}
        else:
            lseats, opp = {1, 2, 3, 4}, {}
        seat = {p: {"rec": [], "logp": [], "val": []} for p in lseats}
        if luck_fn is not None:
            if kind == "d2":
                # mid-game start: baseline is out of distribution -> hard skip
                dealt = {p: None for p in lseats}
            else:
                # capture dealt hands NOW — currentHands mutates as plays occur
                dealt = {p: [int(c) for c in env.game.currentHands[p]] for p in lseats}
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
            if luck_fn is None:
                r[-1] = float(rewards[sp - 1]) / reward_scale
            else:
                raw = float(rewards[sp - 1])
                base = luck_fn(dealt[sp])
                r[-1] = (raw - base) / reward_scale
                if _LUCK_DEBUG["left"] > 0:
                    _LUCK_DEBUG["left"] -= 1
                    print(f"   [luck-baseline] seat {sp}: raw {raw:+.1f} - "
                          f"baseline {base:+.2f} -> shaped {raw - base:+.2f} "
                          f"(scaled {r[-1]:+.4f})")
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
            np.asarray(ret_all, dtype=np.float32),
            mix)


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
               epochs, minibatch, clip, vf_coef, ent_coef, max_grad,
               kl_ref=None, kl_beta=0.0):
    """kl_ref/kl_beta (defaults = exact legacy behavior): anchored RL (M3
    Track C). Adds beta * KL(pi || pi_ref) per decision to the loss, with
    pi_ref a FROZEN reference policy evaluated on the same minibatch. Uses the
    k3 estimator  E_a~mu[ exp(dr) - 1 - dr ],  dr = logp_ref - logp:
    non-negative, exactly 0 when pi == pi_ref, and the gradient flows through
    logp. TODO(M3 Phase 2): adaptive beta (target-KL controller) — fixed beta
    is enough for Phase 1."""
    use_klref = (kl_ref is not None) and (kl_beta > 0.0)
    n = len(records)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    logp_old_t = torch.from_numpy(logp_old).to(device)
    adv_t = torch.from_numpy(adv).to(device)
    ret_t = torch.from_numpy(ret).to(device)
    idx = np.arange(n)
    acc = {"pi": 0.0, "v": 0.0, "ent": 0.0, "kl": 0.0, "klref": 0.0, "nb": 0}
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
            if use_klref:
                with torch.no_grad():
                    logp_ref, _, _ = kl_ref.evaluate(batch)
                dr = logp_ref - logp                       # grad flows via logp
                kl_ref_term = (torch.exp(dr) - 1.0 - dr).mean()
                loss = loss + kl_beta * kl_ref_term
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad)
            opt.step()
            with torch.no_grad():
                acc["pi"] += pi_loss.item()
                acc["v"] += v_loss.item()
                acc["ent"] += ent.item()
                acc["kl"] += (logp_old_t[mb_t] - logp).mean().item()
                if use_klref:
                    acc["klref"] += kl_ref_term.item()
                acc["nb"] += 1
    nb = max(acc["nb"], 1)
    return {k: acc[k] / nb for k in ("pi", "v", "ent", "kl", "klref")}, n


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


def _fit_luck_baseline(path):
    """Fit E[score | dealt hand] on a SNAPSHOT of the online human games
    (M3 Track C --luck-baseline). Reuses ppo/aw_weights.py's fit (mp / n_twos /
    bomb binned means + ridge fallback) verbatim. The live file may be appended
    to by a harvest job mid-read, so we copy it first and tolerate a torn last
    line."""
    import shutil
    import tempfile

    from ppo.aw_weights import fit_baseline, human_seat_rows

    fd, snap = tempfile.mkstemp(suffix=".jsonl", prefix="online_games_snap_")
    os.close(fd)
    try:
        shutil.copyfile(path, snap)
        import json
        games, bad = [], 0
        with open(snap) as f:
            for ln in f:
                try:
                    games.append(json.loads(ln))
                except json.JSONDecodeError:
                    bad += 1                       # torn tail line from harvest
    finally:
        os.unlink(snap)
    rows = human_seat_rows(games)
    base = fit_baseline(rows)
    print(f"[luck-baseline] fit on {len(games)} games ({len(rows)} human-seat rows, "
          f"{bad} torn lines skipped) | r2={base.r2:.3f} mean_score={base.mean_score:+.2f}")
    return base


def _make_luck_fn(base):
    """baseline(dealt hand) -> expected score from deal luck alone.

    IMPORTANT (M3 Phase 1 injection): the baseline is fit on 13-card STARTING
    hands only. For injected D2 MID-GAME episode starts the hand is not a
    fresh 13-card deal — out of distribution — so the baseline is SKIPPED:
    collect() records the dealt hand as None for those episodes (an explicit
    hard skip, NOT a len!=13 heuristic — a mid-game seat can still hold 13
    cards) and luck_fn returns 0.0 for None. The len!=13 guard stays as a
    second line of defense. Injected D1 full deals use the baseline normally."""
    from ppo.aw_weights import has_bomb, min_plays_to_empty, n_twos

    def luck_fn(hand):
        if hand is None or len(hand) != 13:
            return 0.0                             # injected mid-game start
        return float(base.predict(min_plays_to_empty(hand), n_twos(hand),
                                  int(has_bomb(hand))))
    return luck_fn


def launch_gate_kpi(policy, args, update):
    """Save a temp checkpoint and launch the FROZEN gate yardstick
    (eval_policy_gates.py, via ppo/gate_kpi_wrapper.py) on it as a DETACHED
    subprocess — the training loop never blocks on it, and never crashes
    because of it (any failure is logged and swallowed; the wrapper itself
    also appends {"error": ...} lines to the KPI log on gate failure)."""
    try:
        gate_dir = os.path.join(CKPT_DIR, "gate_tmp")
        os.makedirs(gate_dir, exist_ok=True)
        ck = os.path.join(gate_dir, f"ppo_{args.tag}_gate_upd{update:05d}.pt")
        torch.save({"model": policy.state_dict(), "update": update,
                    "arch": args.arch, "args": vars(args)}, ck)
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "gate_kpi_wrapper.py"),
               "--ckpt", ck, "--update", str(update),
               "--games", str(args.gate_games), "--workers", str(args.gate_workers),
               "--log", args.gate_log, "--tag", args.tag]
        with open(ck + ".log", "w") as logf:
            subprocess.Popen(cmd, cwd=_ROOT, stdout=logf, stderr=subprocess.STDOUT,
                             start_new_session=True)
        print(f"   > gate KPI launched (detached) @ upd {update}: "
              f"{args.gate_games} games -> {os.path.basename(args.gate_log)}")
    except Exception as e:                                   # noqa: BLE001
        print(f"   ! gate KPI launch failed (training continues): {e}")


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
    # ── M3 Track C (all default OFF -- defaults preserve legacy behavior exactly) ──
    ap.add_argument("--gate-every", type=int, default=0,
                     help="every N updates, run the frozen gate yardstick (eval_policy_gates.py) "
                          "on a temp checkpoint as a DETACHED subprocess; results append to "
                          "--gate-log as KPI cards (0=off)")
    ap.add_argument("--gate-games", type=int, default=2000,
                     help="games per gate KPI run (frozen card uses 6000; 2000 for speed)")
    ap.add_argument("--gate-workers", type=int, default=2,
                     help="workers for the gate subprocess (machine is shared with the live "
                          "online campaign -- keep total <= 4)")
    ap.add_argument("--gate-log", type=str,
                     default=os.path.join(DATA_DIR, "rl_kpi_log.jsonl"),
                     help="append-only JSONL KPI log written by the gate wrapper")
    ap.add_argument("--luck-baseline", action="store_true",
                     help="subtract E[score | dealt hand] (aw_weights.py fit on the online human "
                          "games snapshot) from the terminal reward at collect time")
    ap.add_argument("--luck-data", type=str,
                     default=os.path.join(DATA_DIR, "online_games.jsonl"),
                     help="online games JSONL the luck baseline is fit on (snapshotted at startup)")
    ap.add_argument("--kl-ref", type=str, default="",
                     help="frozen reference checkpoint for a per-decision KL(pi||pi_ref) loss "
                          "penalty (anchored RL); empty=off")
    ap.add_argument("--kl-beta", type=float, default=0.0,
                     help="coefficient of the KL(pi||pi_ref) penalty (0=off; fixed beta for "
                          "Phase 1, adaptive beta is a TODO in ppo_update)")
    # ── M3 Phase 1: injected episode starts (defaults 0.0 = exact legacy behavior) ──
    ap.add_argument("--inject-d2", type=float, default=0.0,
                     help="fraction of episodes started from a TRAIN-split D2 mid-game "
                          "caged-loss state (big2Game.restore_state; learner forced to the "
                          "record's target seat; luck baseline hard-skipped). 0=off")
    ap.add_argument("--inject-d1", type=float, default=0.0,
                     help="fraction of episodes started from a TRAIN-split D1 strong full "
                          "deal (reset(hands=...); learner at strong_seat; luck baseline "
                          "applies — 13-card hands). 0=off")
    args = ap.parse_args()
    if args.inject_d2 < 0 or args.inject_d1 < 0 or args.inject_d2 + args.inject_d1 > 1.0:
        raise ValueError(f"--inject-d2 + --inject-d1 must be in [0,1] "
                         f"(got {args.inject_d2} + {args.inject_d1})")

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

    # ── M3 Track C setup (all no-ops unless the flags are set) ──
    luck_fn = None
    if args.luck_baseline:
        luck_fn = _make_luck_fn(_fit_luck_baseline(args.luck_data))
        _LUCK_DEBUG["left"] = 6            # print the first few subtractions
    kl_ref = None
    if args.kl_ref:
        kl_ref = _load_frozen(args.kl_ref, args.arch, device)
        print(f"[kl-ref] frozen reference={os.path.basename(args.kl_ref)} beta={args.kl_beta}")
    d2_pool = d1_pool = None
    if args.inject_d2 > 0.0 or args.inject_d1 > 0.0:
        from ppo.injection_pools import load_d1_pool, load_d2_pool
        if args.inject_d2 > 0.0:
            d2_pool = load_d2_pool("train")     # NEVER the eval split
        if args.inject_d1 > 0.0:
            d1_pool = load_d1_pool("train")
        _INJ_DEBUG["left"] = 6                 # print the first few injections
        print(f"[inject] d2 frac={args.inject_d2} pool={len(d2_pool) if d2_pool else 0} "
              f"| d1 frac={args.inject_d1} pool={len(d1_pool) if d1_pool else 0} "
              f"(train split only)")

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
        records, logp_old, adv, ret, inj_mix = collect(
            env, policy, args.games_per_batch, args.gamma, args.lam, args.reward_scale,
            pool=(pool if league else None), learner_seats=args.learner_seats,
            luck_fn=luck_fn,
            inject_d2=args.inject_d2, inject_d1=args.inject_d1,
            d2_pool=d2_pool, d1_pool=d1_pool)
        st, n_steps = ppo_update(policy, opt, records, logp_old, adv, ret, device,
                                 args.epochs, args.minibatch, args.clip,
                                 args.vf_coef, args.ent_coef, args.max_grad,
                                 kl_ref=kl_ref, kl_beta=args.kl_beta)
        dt = time.time() - t0
        klref_col = (f"| klref {st['klref']:.5f} "
                     if (kl_ref is not None and args.kl_beta > 0.0) else "")
        inj_col = (f"| inj nat {inj_mix['natural']} d2 {inj_mix['d2']} d1 {inj_mix['d1']} "
                   if (args.inject_d2 > 0.0 or args.inject_d1 > 0.0) else "")
        print(f"upd {update:3d}/{args.updates} | steps {n_steps:5d} | lr {opt.param_groups[0]['lr']:.2e} "
              f"| pi {st['pi']:+.4f} | v {st['v']:.3f} | ent {st['ent']:.3f} | kl {st['kl']:+.4f} "
              f"{klref_col}{inj_col}| {dt:.1f}s")

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

        # M3 Track C: periodic frozen-gate KPI card (detached; never blocks/crashes)
        if args.gate_every > 0 and update % args.gate_every == 0:
            launch_gate_kpi(policy, args, update)


if __name__ == "__main__":
    main()
