"""Play Big 2 against the trained model from your terminal, and log the results.

You hold ONE seat; the model (default: ppo_cardaware_best.pt) plays the other
three — the mirror of eval_baselines, but with you in the agent's seat. This is
how we start collecting the real signal that actually matters: avg_score vs a
human (see SURVEY_evaluation.md). Your per-game score (and a running avg_score +
win-rate) is printed and appended to a log file.

Run from the worktree root:
    ./.venv/bin/python -m ppo.play_human                 # interactive
    ./.venv/bin/python -m ppo.play_human --games 20
    ./.venv/bin/python -m ppo.play_human --auto 50       # self-test: Smart plays your seat

Controls: type the number of the action you want, or `p` to pass, `q` to quit.
"""
import argparse
import json
import os

import numpy as np
import torch

import enumerateOptions as _eo
import gameLogic
from engine.env import Big2Env, PASS_IDX
from ppo.policies import make_policy
from ppo.eval_baselines import smart_action  # for --auto self-test

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['♦', '♣', '♥', '♠']
DEFAULT_CKPT = os.path.join(os.path.dirname(__file__), "checkpoints", "ppo_cardaware_best.pt")
DEFAULT_LOG = os.path.join(os.path.dirname(__file__), "human_games.jsonl")


def card_str(cid):
    cid = int(cid)
    return f"{RANKS[(cid - 1) // 4]}{SUITS[(cid - 1) % 4]}"


def cards_str(cards):
    return ' '.join(card_str(c) for c in sorted(int(c) for c in cards))


def five_kind(cards):
    arr = np.array(cards)
    if gameLogic.isStraightFlush(arr):
        return "同花順"
    if gameLogic.isFourOfAKind(arr):
        return "鐵支"
    if gameLogic.isFullHouse(arr)[0]:
        return "葫蘆"
    return "順子"


def describe_action(a):
    if a == PASS_IDX:
        return "PASS"
    cards, nc = _eo.getOptionNC(a)
    if nc == 1:
        return f"單張 {cards_str(cards)}"
    if nc == 2:
        return f"對子 {cards_str(cards)}"
    return f"五張({five_kind(cards)}) {cards_str(cards)}"


def show_state(env, human_seat):
    g = env.game
    print("\n" + "─" * 56)
    if g.control == 1:
        print("檯面:(你有控制權,可任意出牌)")
    else:
        trick = [int(c) for c in g.handsPlayed[g.goIndex - 1].hand]
        print(f"檯面要壓過:{cards_str(trick)}")
    counts = " | ".join(
        f"{'你' if p == human_seat else 'AI'}P{p}:{g.currentHands[p].size}張"
        for p in range(1, 5))
    print(f"剩牌數  {counts}    (本輪 pass {g.passCount})")
    print(f"你的手牌:{cards_str(g.currentHands[human_seat])}")


def human_action(env, human_seat):
    show_state(env, human_seat)
    mask = env.get_valid_actions()
    legal = [int(a) for a in np.flatnonzero(mask)]
    non_pass = sorted(a for a in legal if a != PASS_IDX)
    pass_ok = bool(mask[PASS_IDX])
    if not non_pass:
        print("(無牌可出,自動 PASS)")
        return PASS_IDX
    print("可出:")
    for i, a in enumerate(non_pass):
        print(f"  [{i}] {describe_action(a)}")
    if pass_ok:
        print("  [p] PASS")
    while True:
        s = input("選擇> ").strip().lower()
        if s == 'q':
            return 'quit'
        if s == 'p' and pass_ok:
            return PASS_IDX
        if s.isdigit() and 0 <= int(s) < len(non_pass):
            return non_pass[int(s)]
        print("  輸入無效,再試一次(數字 / p / q)")


def play(args):
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    policy = make_policy(ck["arch"], device)
    policy.net.load_state_dict(ck["model"])
    policy.eval()
    print(f"對手 model:{os.path.basename(args.checkpoint)} "
          f"(arch={ck['arch']}, upd={ck.get('update')}, device={device})")

    env = Big2Env()
    scores, wins, g = [], 0, 0
    while args.games == 0 or g < args.games:
        human_seat = (g % 4) + 1
        env.reset()
        print(f"\n=== 第 {g + 1} 局 — 你是 P{human_seat} ===")
        quit_now = False
        while not env.done:
            p = env.current_player
            if p == human_seat:
                if args.auto:
                    a = smart_action(env)
                else:
                    a = human_action(env, human_seat)
                    if a == 'quit':
                        quit_now = True
                        break
            else:
                a = policy.greedy(env)
            rewards, _done = env.step(a)
        if quit_now:
            print("\n結束。")
            break

        sc = float(rewards[human_seat - 1])
        scores.append(sc)
        wins += int(sc > 0)
        g += 1
        avg = sum(scores) / len(scores)
        print(f"\n本局你的分數:{sc:+.0f}   |   "
              f"累計 {g} 局:勝率 {wins/g*100:.0f}%  平均分 {avg:+.2f}")
        with open(args.log, "a") as f:
            f.write(json.dumps({
                "game": g, "human_seat": human_seat,
                "scores": [float(x) for x in rewards],
                "human_score": sc, "checkpoint": os.path.basename(args.checkpoint),
                "ckpt_update": ck.get("update"),
            }) + "\n")

    if scores:
        avg = sum(scores) / len(scores)
        print("\n" + "=" * 56)
        print(f"總結:{len(scores)} 局 | 你的勝率 {wins/len(scores)*100:.1f}% | "
              f"你的平均分 {avg:+.2f}  (model 約為 {-avg:+.2f})")
        print(f"紀錄已存到 {args.log}")
        if not args.auto:
            print("提示:大老二單局方差大,要可信至少打 ~100 局。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--games", type=int, default=0, help="0 = 一直玩到你按 q")
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--auto", type=int, default=0, help="自我測試:Smart 代打你的座位 N 局")
    args = ap.parse_args()
    if args.auto:
        args.games = args.auto
    play(args)


if __name__ == "__main__":
    main()
