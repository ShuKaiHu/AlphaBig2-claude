"""FROZEN gate-metric evaluator for pure policies (plan P0, PLAN_policy_score_signal.md).

Every AW-BC candidate (and policy_4500 itself, for the baseline card) is graded by
EXACTLY this yardstick -- do not reinterpret the definitions here without bumping
the plan.

Evaluation game setup (paired across candidates):
  * engine big2Game via engine.env.Big2Env
  * CANDIDATE holds one seat, rotated by game index: seat = (g % 4) + 1
  * the other 3 seats are the FIXED incumbent policy_4500_best (argmax)
  * deal g is dealt with np.random.seed(seed_base + g) -> every future candidate
    faces the IDENTICAL (deal, seat) sequence -> paired comparison
  * candidate acts by argmax (deployment mode)

Gate metrics (candidate seat only):
  G1a  mp<=6 win rate, bomb-free stratum        (win = candidate empties first)
  G1b  free-lead 5-card rate, bomb-free mp<=5   (free lead = control==1, incl. 3C opening)
  G1c  intact-combo rate, bomb-free mp<=5       (greedy-decomposition 5-card combos
                                                 played intact in a single play)
  G2   hoarding: P(>=1 two unplayed | lost with remaining>=10 and dealt >=1 two)
  G4a  avg_score over ALL games (engine scoring)
  G4b  humanness: measure_pool_humanness.py methodology (greedy top-1 match on the
       human decisions of a FROZEN online_games snapshot, bc_dataset target="human")

Online human anchors for DIRECTION (not offline targets): G1b 47.4%, G1c 59.7%, G2 57.4%.

Usage:
    ./.venv/bin/python3 eval_policy_gates.py --ckpt ppo/checkpoints/policy_4500_best.pt \
        --games 6000 --workers 6 --seed-base 0 --out ppo/data/gates_baseline_policy4500.json
    ./.venv/bin/python3 eval_policy_gates.py --ckpt ... --trace 3   # verbose single game
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import enumerateOptions as eo
from big2Game import big2Game
from engine.env import Big2Env, PASS_IDX

# ── min-plays / greedy decomposition: the SAME code the value line uses ──────
_HAND_FEATURES_PATH = os.path.join(_HERE, "value", "hand_features.py")

_spec = importlib.util.spec_from_file_location("value_hand_features", _HAND_FEATURES_PATH)
_HF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_HF)
min_plays_to_empty = _HF.min_plays_to_empty
_extract_five = _HF._extract_five

DEFAULT_OPP = os.path.join(_HERE, "ppo", "checkpoints", "policy_4500_best.pt")
LIVE_DATA = os.path.join(_HERE, "ppo", "data", "online_games.jsonl")
FROZEN_HUMANNESS_DATA = os.path.join(_HERE, "ppo", "data", "online_games_gates_frozen.jsonl")


def greedy_five_combos(cards):
    """The 5-card combos of hand_features' greedy decomposition, in extraction
    order. Prefix-identical to min_plays_to_empty (same _extract_five loop)."""
    rem = [int(c) for c in cards]
    fives = []
    while True:
        five = _extract_five(rem)
        if not five:
            break
        fives.append(sorted(int(c) for c in five))
        for c in five:
            rem.remove(c)
    return fives


def is_bomb_hand(cards):
    """Bomb in a starting hand = four-of-a-kind rank OR 5 same-suit cards on one
    of the 10 legal straight windows. Delegates to the ENGINE's own detectors
    (gameLogic._STRAIGHT_SEQUENCES has exactly those 10 windows; J-Q-K-A-2 is
    not one of them)."""
    arr = np.asarray([int(c) for c in cards])
    return bool(big2Game._has_four_of_a_kind(None, arr)
                or big2Game._has_straight_flush(None, arr))


def _n_twos(cards):
    return sum(1 for c in cards if (int(c) - 1) // 4 == 12)


def load_net(path, device="cpu"):
    import torch
    from ppo.network_cardaware import CardAwareActorCritic
    ck = torch.load(path, map_location=device, weights_only=False)
    arch = ck.get("arch", "cardaware")
    if arch != "cardaware":
        raise ValueError(f"{path}: arch={arch!r}; this evaluator is pure-policy "
                         f"cardaware only (plan scope)")
    net = CardAwareActorCritic().to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    return net


def _fmt_card(c):
    faces = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2"]
    suits = ["C", "D", "H", "S"]
    c = int(c)
    return faces[(c - 1) // 4] + suits[(c - 1) % 4]


def play_game(g, cand_net, opp_net, seed_base, trace=False):
    """Roll out fixed-deal game g; candidate in seat (g%4)+1, incumbent argmax
    in the other three. Returns the per-game gate record."""
    np.random.seed(seed_base + g)
    env = Big2Env()          # big2Game() constructor deals with the seeded RNG
    game = env.game
    seat = (g % 4) + 1
    deal = {s: [int(c) for c in game.currentHands[s]] for s in range(1, 5)}
    deal_md5 = hashlib.md5(json.dumps(deal, sort_keys=True).encode()).hexdigest()
    start = deal[seat]
    if trace:
        print(f"game {g} seed {seed_base + g} cand_seat {seat} deal_md5 {deal_md5}")
        for s in range(1, 5):
            tag = " <- CANDIDATE" if s == seat else ""
            print(f"  seat {s}: {' '.join(_fmt_card(c) for c in deal[s])}{tag}")

    decisions = []
    rewards = None
    while not env.done:
        p = env.current_player
        legal = np.flatnonzero(env.get_valid_actions())
        if len(legal) == 0:
            a = PASS_IDX                      # defensive; engine skips passed seats
        elif len(legal) == 1:
            a = int(legal[0])                 # forced -> same as argmax, skip forward
        else:
            net = cand_net if p == seat else opp_net
            a = net.greedy(env, "cpu")
        free_lead = bool(game.control == 1)
        if a == PASS_IDX:
            cards = []
        else:
            cards, _nc = eo.getOptionNC(a)
            cards = [int(c) for c in cards]
        if p == seat:
            decisions.append({"free_lead": free_lead, "cards": cards})
        if trace:
            who = "CAND" if p == seat else "opp "
            what = "PASS" if not cards else " ".join(_fmt_card(c) for c in cards)
            print(f"  [{'lead' if free_lead else 'follow'}] seat {p} {who}: {what}"
                  f"  (hand left after: {game.currentHands[p].size - len(cards)})")
        rewards, done = env.step(a)

    final_rem = int(game.currentHands[seat].size)
    rec = {
        "g": g,
        "seat": seat,
        "deal_md5": deal_md5,
        "start_hand": start,
        "mp": int(min_plays_to_empty(start)),
        "bomb": is_bomb_hand(start),
        "five_combos": greedy_five_combos(start),
        "dealt_twos": _n_twos(start),
        "decisions": decisions,
        "final_remaining": final_rem,
        "unplayed_twos": _n_twos(game.currentHands[seat]),
        "score": float(rewards[seat - 1]),
        "win": final_rem == 0,
        "rewards": [float(r) for r in rewards],
    }
    if trace:
        print(f"  final: win={rec['win']} remaining={final_rem} "
              f"score={rec['score']:+.0f} rewards={rec['rewards']} "
              f"(sum {sum(rec['rewards']):.0f})")
        print(f"  start mp={rec['mp']} bomb={rec['bomb']} "
              f"combos={[[_fmt_card(c) for c in f] for f in rec['five_combos']]}")
    return rec


# ── multiprocessing plumbing ────────────────────────────────────────────────
_W = {}


def _worker_init(cand_path, opp_path, seed_base):
    import torch
    torch.set_num_threads(1)
    cand = load_net(cand_path, "cpu")
    opp = cand if os.path.abspath(opp_path) == os.path.abspath(cand_path) \
        else load_net(opp_path, "cpu")
    _W.update(cand=cand, opp=opp, seed_base=seed_base)


def _worker_game(g):
    return play_game(g, _W["cand"], _W["opp"], _W["seed_base"])


def run_games(cand_path, opp_path, n_games, workers, seed_base):
    t0 = time.time()
    if workers <= 1:
        _worker_init(cand_path, opp_path, seed_base)
        recs = [_worker_game(g) for g in range(n_games)]
    else:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(workers, initializer=_worker_init,
                      initargs=(cand_path, opp_path, seed_base)) as pool:
            recs = []
            for i, r in enumerate(pool.imap_unordered(_worker_game, range(n_games),
                                                      chunksize=16)):
                recs.append(r)
                if (i + 1) % 500 == 0:
                    el = time.time() - t0
                    print(f"  {i+1}/{n_games} games  ({el:.0f}s, "
                          f"{(i+1)/el:.1f} games/s)", flush=True)
    recs.sort(key=lambda r: r["g"])
    print(f"[games] {n_games} games in {time.time()-t0:.0f}s")
    return recs


# ── metrics ─────────────────────────────────────────────────────────────────
def wilson(k, n, z=1.96):
    if n == 0:
        return {"n": 0, "k": 0, "rate": None, "ci95": [None, None]}
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"n": n, "k": k, "rate": p, "ci95": [c - hw, c + hw]}


def compute_metrics(recs):
    m = {}
    n = len(recs)
    m["n_games"] = n

    # G1a: bomb-free mp<=6 win rate
    s = [r for r in recs if r["mp"] <= 6 and not r["bomb"]]
    m["G1a_mp6_win_rate"] = wilson(sum(r["win"] for r in s), len(s))

    # G1b/G1c stratum: bomb-free mp<=5
    s5 = [r for r in recs if r["mp"] <= 5 and not r["bomb"]]
    m["G1bc_stratum_games"] = len(s5)
    leads = [d for r in s5 for d in r["decisions"] if d["free_lead"]]
    m["G1b_lead_5card_rate"] = wilson(
        sum(1 for d in leads if len(d["cards"]) == 5), len(leads))

    combos_total, combos_intact = 0, 0
    for r in s5:
        plays5 = {frozenset(d["cards"]) for d in r["decisions"] if len(d["cards"]) == 5}
        for combo in r["five_combos"]:
            combos_total += 1
            combos_intact += int(frozenset(combo) in plays5)
    m["G1c_intact_combo_rate"] = wilson(combos_intact, combos_total)

    # G2: hoarding twos in caged losses
    s2 = [r for r in recs
          if (not r["win"]) and r["final_remaining"] >= 10 and r["dealt_twos"] >= 1]
    m["G2_hoard_two_rate"] = wilson(
        sum(1 for r in s2 if r["unplayed_twos"] >= 1), len(s2))

    # G4a: avg score over ALL games
    scores = np.array([r["score"] for r in recs], dtype=np.float64)
    se = float(scores.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    m["G4a_avg_score"] = {
        "n": n, "mean": float(scores.mean()),
        "ci95": [float(scores.mean() - 1.96 * se), float(scores.mean() + 1.96 * se)],
        "std": float(scores.std(ddof=1)) if n > 1 else None,
    }
    m["overall_win_rate"] = wilson(sum(r["win"] for r in recs), n)
    return m


# ── G4b humanness (measure_pool_humanness.py methodology, frozen snapshot) ──
def ensure_frozen_snapshot(src=LIVE_DATA, dst=FROZEN_HUMANNESS_DATA):
    """First call freezes the current online_games.jsonl (complete JSON lines
    only -- a harvest job may be appending a partial last line). Later runs
    reuse the SAME frozen file so G4b is comparable across candidates."""
    if os.path.exists(dst):
        return dst
    good = []
    with open(src, "rb") as f:
        for ln in f.read().split(b"\n"):
            if not ln.strip():
                continue
            try:
                json.loads(ln)
                good.append(ln)
            except json.JSONDecodeError:
                pass
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b"\n".join(good) + b"\n")
    os.replace(tmp, dst)
    print(f"[humanness] froze {len(good)} games -> {dst}")
    return dst


def measure_humanness(ckpt_path, data_path, device=None):
    """Fraction of human decisions (bc_dataset target='human' on the frozen
    snapshot) where the candidate's argmax matches the human play -- exactly
    measure_pool_humanness.py's metric for 'cardaware' members."""
    import torch
    import ppo.bc_dataset as bc
    from ppo.network_cardaware import build_batch
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    old = bc.DATA
    bc.DATA = data_path
    try:
        recs = bc.build_records(target="human")
    finally:
        bc.DATA = old
    net = load_net(ckpt_path, device)
    correct = 0
    with torch.no_grad():
        for s in range(0, len(recs), 512):
            b = build_batch(recs[s:s + 512], device)
            logits, _ = net.forward(b["obs"], b["act_feats"], b["act_mask"])
            correct += (logits.argmax(-1) == b["pos"]).sum().item()
    return correct / max(len(recs), 1), len(recs)


# ── card printing ───────────────────────────────────────────────────────────
def _pct(d):
    if d["rate"] is None:
        return "  n/a"
    return (f"{d['rate']*100:5.1f}%  [{d['ci95'][0]*100:.1f}, {d['ci95'][1]*100:.1f}]"
            f"  (k={d['k']}, n={d['n']})")


def print_card(m, ckpt, anchors=True):
    print(f"\n=== policy gate card: {ckpt} ===")
    print(f"games: {m['n_games']}")
    print(f"G1a  mp<=6 win rate (bomb-free):     {_pct(m['G1a_mp6_win_rate'])}")
    print(f"G1b  lead-5card rate (mp<=5, b-f):   {_pct(m['G1b_lead_5card_rate'])}")
    print(f"G1c  intact-combo rate (mp<=5, b-f): {_pct(m['G1c_intact_combo_rate'])}")
    print(f"G2   hoard-2 rate (caged losses):    {_pct(m['G2_hoard_two_rate'])}")
    g4a = m["G4a_avg_score"]
    print(f"G4a  avg score (all games):          {g4a['mean']:+.3f}  "
          f"[{g4a['ci95'][0]:+.3f}, {g4a['ci95'][1]:+.3f}]")
    print(f"     overall win rate:               {_pct(m['overall_win_rate'])}")
    if "G4b_humanness" in m:
        h = m["G4b_humanness"]
        print(f"G4b  humanness (frozen snapshot):    {h['rate']*100:.1f}%  (n={h['n']})")
    if anchors:
        print("     [online human anchors, direction only: "
              "G1b 47.4% / G1c 59.7% / G2 57.4%]")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--out", default=None, help="card JSON path")
    ap.add_argument("--opp", default=DEFAULT_OPP,
                    help="fixed incumbent opponents (default policy_4500_best)")
    ap.add_argument("--humanness-data", default=FROZEN_HUMANNESS_DATA)
    ap.add_argument("--skip-humanness", action="store_true")
    ap.add_argument("--pergame-out", default=None,
                    help="gzipped per-game records (default <out>.pergame.json.gz)")
    ap.add_argument("--trace", type=int, default=None,
                    help="print a full trace of ONE game index and exit")
    args = ap.parse_args()
    args.workers = min(args.workers, 6)   # online campaign shares this machine

    if args.trace is not None:
        cand = load_net(args.ckpt, "cpu")
        opp = cand if os.path.abspath(args.opp) == os.path.abspath(args.ckpt) \
            else load_net(args.opp, "cpu")
        play_game(args.trace, cand, opp, args.seed_base, trace=True)
        return

    recs = run_games(args.ckpt, args.opp, args.games, args.workers, args.seed_base)
    metrics = compute_metrics(recs)

    if not args.skip_humanness:
        data = ensure_frozen_snapshot(dst=args.humanness_data)
        rate, n = measure_humanness(args.ckpt, data)
        metrics["G4b_humanness"] = {"rate": rate, "n": n, "data": data}

    print_card(metrics, args.ckpt)

    card = {
        "ckpt": os.path.abspath(args.ckpt),
        "opp": os.path.abspath(args.opp),
        "games": args.games,
        "seed_base": args.seed_base,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "deal_md5_first3": [r["deal_md5"] for r in recs[:3]],  # pairing fingerprint
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(card, f, indent=1)
        print(f"[out] card -> {args.out}")
        pergame = args.pergame_out or args.out.replace(".json", "") + ".pergame.json.gz"
        with gzip.open(pergame, "wt") as f:
            json.dump(recs, f)
        print(f"[out] per-game records -> {pergame}")


if __name__ == "__main__":
    main()
