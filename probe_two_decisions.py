"""Decision-level 2-dumping probe (user-specified methodology, 2026-07-19).

The end-of-game hoard metric conflates opportunity, forced play, and downstream
dynamics. This probe measures the DECISION itself: at every real-human-game state
where the target seat (a) is on the caged-loss track, (b) holds >= 1 two, and
(c) has BOTH a legal action that plays a 2 AND a legal alternative that doesn't
(a genuine choice), ask each checkpoint what it would do.

Metrics per checkpoint, paired per decision point:
  - P(argmax action plays a 2 | opportunity)
  - mean probability mass on 2-playing legal actions
  - flip counts vs baseline (no->yes / yes->no), the user's requested number.

Population = same as the D2 pool: seats that ENDED as caged losses (final
remaining >= 10, negative score) having been dealt >= 1 two. Window = every
decision from the first event where ANY opponent is at <= 8 cards. Replay uses
the bc_dataset-verified trick logic (a play does NOT reopen the trick; only 3
CONSECUTIVE passes reset it, server autos included). "Plays a 2" is decided by
the ENGINE (clone -> step -> hand diff), never by decoding action tables.

    PYTHONPATH=. .venv/bin/python3 probe_two_decisions.py \
        --ckpts ppo/checkpoints/policy_4500_best.pt ppo/checkpoints/ppo_m3p1_best.pt \
        --out ppo/data/probe_two_m3p1.json
"""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from big2Game import big2Game                                   # noqa: E402
from engine.env import Big2Env                                  # noqa: E402
from ppo.network_cardaware import obs_from_env, legal_action_data, _to_batch  # noqa: E402
from eval_injected_d2 import load_frozen_policy                 # noqa: E402

# md5 f23cc53c... == the frozen Phase-0 snapshot (verified identical); master has
# not grown since (harvest idle), so split hashing matches the pools exactly.
SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "ppo", "data", "online_games.jsonl")
CAGE, DANGER = 10, 8


def game_split(gid):
    return "eval" if int(hashlib.md5(gid.encode()).hexdigest(), 16) % 10 == 0 else "train"


def legal_probs(pol, env):
    """(legal_idx, softmax probs) — mirrors network_cardaware.greedy's forward."""
    net, device = pol.net, pol.device
    obs = _to_batch([obs_from_env(env)], device)
    legal_idx, feats = legal_action_data(env)
    af = torch.from_numpy(feats).unsqueeze(0).to(device)
    am = torch.ones(1, len(legal_idx), dtype=torch.bool, device=device)
    with torch.no_grad():
        logits, _ = net.forward(obs, af, am)
        p = torch.softmax(logits[0], dim=-1).cpu().numpy()
    return legal_idx, p


def action_plays_two(base_game, action_idx):
    """Engine-decided: clone the state, apply the action, diff the mover's hand."""
    g = copy.deepcopy(base_game)
    env = Big2Env.__new__(Big2Env)
    env._game, env._done = g, False
    mover = g.playersGo
    before = set(int(c) for c in g.currentHands[mover])
    env.step(int(action_idx))
    after = set(int(c) for c in g.currentHands[mover])
    played = before - after
    return any(c >= 49 for c in played), len(played) > 0   # (plays_two, was_a_play)


def restored(hands, turn, trick_cards, trick_owner, passed, cards_played):
    return big2Game.restore_state(
        hands={s: list(hands[s]) for s in range(4)},
        whose_turn=turn + 1,
        trick_cards=(list(trick_cards) if trick_cards else None),
        trick_owner=(trick_owner + 1) if trick_cards else None,
        passed_players=[s + 1 for s in passed],
        cards_played={s: list(cards_played[s]) for s in range(4)},
        must_play_club3=False,
    )


def decision_points(games, limit=0):
    """Yield probe states via the bc_dataset-verified replay loop."""
    n_pts = 0
    for g in games:
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        scores = {int(k): float(v) for k, v in g["scores"].items()}
        our = int(g.get("our_seat", -1))
        targets = set()
        for s in range(4):
            dealt2 = sum(1 for c in hands[s] if c >= 49)
            # final remaining inferred at end of replay; pre-filter on dealt 2 + loss
            if dealt2 >= 1 and scores.get(s, 0) < 0:
                targets.add(s)
        if not targets:
            continue
        remaining = {s: list(hands[s]) for s in range(4)}
        cards_played = {s: [] for s in range(4)}
        trick_cards, trick_owner, passed, consec = None, None, set(), 0
        events = g["plays"]
        # final remaining per seat (to keep only true caged losers)
        fin = {s: len(hands[s]) - sum(len(e["cards"]) for e in events
                                      if e["seat"] == s and e["action"] == "play")
               for s in range(4)}
        targets = {s for s in targets if fin[s] >= CAGE}
        if not targets:
            continue
        for ev in events:
            s, act = ev["seat"], ev["action"]
            cards = list(map(int, ev["cards"]))
            danger = any(len(remaining[o]) <= DANGER for o in range(4) if o not in targets)
            if (s in targets and danger and act != "pass" or
                    s in targets and danger and act == "pass" and s not in passed):
                holds2 = any(c >= 49 for c in remaining[s])
                if holds2:
                    try:
                        base = restored(remaining, s, trick_cards, trick_owner,
                                        passed, cards_played)
                        env = Big2Env.__new__(Big2Env)
                        env._game, env._done = base, False
                        legal, _ = legal_action_data(env)
                        flags = [action_plays_two(base, a) for a in legal]
                        two_opts = [i for i, (t, p) in enumerate(flags) if t]
                        alt_opts = [i for i, (t, p) in enumerate(flags) if not t]
                        if two_opts and alt_opts:
                            n_pts += 1
                            human2 = act == "play" and any(c >= 49 for c in cards)
                            yield dict(gid=g.get("id"), split=game_split(g.get("id", "")),
                                       seat=s, is_bot=(s == our), base=base, legal=legal,
                                       two_pos=set(two_opts), human_plays2=human2)
                            if limit and n_pts >= limit:
                                return
                    except Exception:
                        pass   # unreconstructable state: skip, count nothing
            # apply event (verified trick logic)
            if act == "play":
                for c in cards:
                    remaining[s].remove(c)
                    cards_played[s].append(c)
                trick_cards, trick_owner = cards, s
                passed.discard(s)
                consec = 0
            else:
                if s not in passed:
                    passed.add(s)
                consec += 1
                if consec >= 3:
                    trick_cards, trick_owner, passed, consec = None, None, set(), 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="first = baseline, rest = candidates (paired vs baseline)")
    ap.add_argument("--data", default=SNAPSHOT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--mode", default="all", choices=["all", "last-chance"])
    args = ap.parse_args()

    if args.mode == "last-chance":
        run_last_chance(args)
        return

    device = "cpu"
    torch.set_num_threads(2)
    pols = [load_frozen_policy(p, device) for p in args.ckpts]
    names = [os.path.basename(p) for p in args.ckpts]

    games = [json.loads(l) for l in open(args.data)]
    rows = []
    for pt in decision_points(games, args.limit):
        env = Big2Env.__new__(Big2Env)
        env._game, env._done = pt["base"], False
        rec = {"gid": pt["gid"], "split": pt["split"], "is_bot": pt["is_bot"],
               "human_plays2": pt["human_plays2"]}
        for nm, pol in zip(names, pols):
            legal, probs = legal_probs(pol, env)
            assert list(legal) == list(pt["legal"])
            arg = int(np.argmax(probs))
            rec[nm] = {"argmax_plays2": arg in pt["two_pos"],
                       "mass2": float(sum(probs[i] for i in pt["two_pos"]))}
        rows.append(rec)

    print(f"decision points with a GENUINE 2-play choice: {len(rows)} "
          f"(train {sum(1 for r in rows if r['split']=='train')} / "
          f"eval {sum(1 for r in rows if r['split']=='eval')})")
    base = names[0]
    for split in ("純真人座位", "bot座位", "all"):
        if split == "純真人座位": sub = [r for r in rows if not r["is_bot"]]
        elif split == "bot座位": sub = [r for r in rows if r["is_bot"]]
        else: sub = rows
        if not sub:
            continue
        print(f"--- split={split} (n={len(sub)}) ---")
        print(f"  {'真人實際行為':26} P(出2)={100*np.mean([r['human_plays2'] for r in sub]):5.1f}%")
        for nm in names:
            pa = np.mean([r[nm]["argmax_plays2"] for r in sub])
            pm = np.mean([r[nm]["mass2"] for r in sub])
            line = f"  {nm:28} P(argmax出2)={100*pa:5.1f}%  出2機率質量={100*pm:5.1f}%"
            if nm != base:
                f_ny = sum(1 for r in sub if not r[base]["argmax_plays2"] and r[nm]["argmax_plays2"])
                f_yn = sum(1 for r in sub if r[base]["argmax_plays2"] and not r[nm]["argmax_plays2"])
                dm = np.mean([r[nm]["mass2"] - r[base]["mass2"] for r in sub])
                line += f"  翻轉 不出→出:{f_ny} 出→不出:{f_yn}  Δ質量={100*dm:+.2f}pp"
            print(line)
    if args.out:
        json.dump({"ckpts": args.ckpts, "n": len(rows), "rows": rows}, open(args.out, "w"))
        print(f"wrote {args.out}")




# ═══════════════ last-chance mode (user-frozen spec, 2026-07-19 plan) ═══════════════
# Population: HUMAN seats only (exclude our_seat), losers (score<0) dealt >=1 two.
# Opportunity: >=1 legal action plays a 2 (engine's own action tables; forced points
# count). Judged at the seat's chronologically LAST opportunity point. Multi-2 edge:
# judged at that last point (declining a later chance for a second 2 = 沒出).
import enumerateOptions as eo

PLAYS2 = np.zeros(eo.passInd + 1, dtype=bool)
for _i, _cid in enumerate(eo.SINGLE_ACTIONS):
    PLAYS2[_i] = _cid >= 49
for _i, _pr in enumerate(eo.PAIR_ACTIONS):
    PLAYS2[eo.PAIR_OFFSET + _i] = any(c >= 49 for c in _pr)
for _i, _fv in enumerate(eo.FIVE_ACTIONS):
    PLAYS2[eo.FIVE_OFFSET + _i] = any(c >= 49 for c in _fv)
assert not PLAYS2[eo.passInd]


def last_chance_points(games):
    """One lightweight spec per qualifying seat: its LAST legal-2-play point."""
    specs = {}
    for g in games:
        our = int(g.get("our_seat", -1))
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        scores = {int(k): float(v) for k, v in g["scores"].items()}
        targets = {s for s in range(4)
                   if s != our and scores.get(s, 0) < 0
                   and any(c >= 49 for c in hands[s])}
        if not targets:
            continue
        events = g["plays"]
        fin = {s: len(hands[s]) - sum(len(e["cards"]) for e in events
                                      if e["seat"] == s and e["action"] == "play")
               for s in range(4)}
        remaining = {s: list(hands[s]) for s in range(4)}
        cards_played = {s: [] for s in range(4)}
        trick_cards, trick_owner, passed, consec = None, None, set(), 0
        later = {}   # (gid,seat)-key -> decisions seen after current last point
        for ev in events:
            s, act = ev["seat"], ev["action"]
            cards = list(map(int, ev["cards"]))
            is_decision = not (act == "pass" and s in passed)   # server auto-pass ≠ decision
            if s in targets and is_decision:
                key = (g.get("id"), s)
                if key in specs:
                    specs[key]["later_decisions"] += 1
                if any(c >= 49 for c in remaining[s]):
                    try:
                        base = restored(remaining, s, trick_cards, trick_owner,
                                        passed, cards_played)
                        env = Big2Env.__new__(Big2Env)
                        env._game, env._done = base, False
                        legal, _ = legal_action_data(env)
                        two_legal = [a for a in legal if PLAYS2[a]]
                        if two_legal:
                            specs[key] = dict(
                                gid=g.get("id"), split=game_split(g.get("id", "")),
                                seat=s, caged=fin[s] >= CAGE,
                                forced=all(PLAYS2[a] for a in legal),
                                human_plays2=(act == "play" and any(c >= 49 for c in cards)),
                                remaining={k: list(v) for k, v in remaining.items()},
                                cards_played={k: list(v) for k, v in cards_played.items()},
                                trick_cards=(list(trick_cards) if trick_cards else None),
                                trick_owner=trick_owner,
                                passed=set(passed), later_decisions=0,
                                n_legal=len(legal), n_two_legal=len(two_legal))
                    except Exception:
                        pass
            if act == "play":
                for c in cards:
                    remaining[s].remove(c)
                    cards_played[s].append(c)
                trick_cards, trick_owner = cards, s
                passed.discard(s)
                consec = 0
            else:
                if s not in passed:
                    passed.add(s)
                consec += 1
                if consec >= 3:
                    trick_cards, trick_owner, passed, consec = None, None, set(), 0
    return list(specs.values())


def rebuild(spec):
    base = restored(spec["remaining"], spec["seat"], spec["trick_cards"],
                    spec["trick_owner"], spec["passed"], spec["cards_played"])
    env = Big2Env.__new__(Big2Env)
    env._game, env._done = base, False
    return env


def run_last_chance(args):
    device = "cpu"
    torch.set_num_threads(2)
    pols = [load_frozen_policy(p, device) for p in args.ckpts]
    names = [os.path.basename(p) for p in args.ckpts]
    games = [json.loads(l) for l in open(args.data)]
    specs = last_chance_points(games)
    print(f"母體(輸家+發到2+非our_seat)最後機會點:{len(specs)} 個")

    # 交叉驗證:動作表 PLAYS2 vs 計畫指定的 clone→step→手牌差分,500 隨機 (點,動作)
    import random
    random.seed(0)
    ok = 0
    for spec in random.sample(specs, min(120, len(specs))):
        env = rebuild(spec)
        legal, _ = legal_action_data(env)
        for a in random.sample(list(legal), min(4, len(legal))):
            t_table = bool(PLAYS2[a])
            t_diff, _ = action_plays_two(env.game, a)
            assert t_table == t_diff, f"表/差分不一致 action={a}"
            ok += 1
    print(f"交叉驗證:動作表 vs clone-diff 一致 {ok}/{ok} ✅")

    rows = []
    for spec in specs:
        env = rebuild(spec)
        rec = {k: spec[k] for k in ("gid", "split", "seat", "caged", "forced",
                                    "human_plays2", "later_decisions")}
        for nm, pol in zip(names, pols):
            legal, probs = legal_probs(pol, env)
            arg = int(np.argmax(probs))
            rec[nm] = {"argmax_plays2": bool(PLAYS2[legal[arg]]),
                       "mass2": float(sum(p for i, p in zip(legal, probs) if PLAYS2[i]))}
        rows.append(rec)

    def table(sub, label):
        if not sub:
            return
        print(f"--- {label} (n={len(sub)}) ---")
        print(f"  {'真人實際':26} P(出2)={100*np.mean([r['human_plays2'] for r in sub]):5.1f}%")
        for nm in names:
            pa = np.mean([r[nm]["argmax_plays2"] for r in sub])
            pm = np.mean([r[nm]["mass2"] for r in sub])
            print(f"  {nm:26} P(argmax出2)={100*pa:5.1f}%  質量={100*pm:5.1f}%")

    print(f"\n被迫點占比:{100*np.mean([r['forced'] for r in rows]):.1f}%")
    table(rows, "主表:全部最後機會點(含被迫,依規格)")
    table([r for r in rows if not r["forced"]], "附註:有選擇的點")
    table([r for r in rows if r["caged"]], "附註:被關子集(rem>=10)")
    table([r for r in rows if r["split"] == "eval"], "附註:eval split(m3p1 未見過)")
    if args.out:
        slim = [{k: v for k, v in r.items()} for r in rows]
        json.dump({"ckpts": args.ckpts, "n": len(rows), "rows": slim},
                  open(args.out, "w"), default=str)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
