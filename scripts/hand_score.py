#!/usr/bin/env python3
"""手牌強度評分 CLI(v2 加權指數推論;只需 numpy,不需 sklearn)。

分數含義(權重學自 6,225 局的真人三家,見 reports/hand_score_v2.md):
  exp_score = 這手牌在真人手上的期望得分(ridge)
  win_prob  = 真人拿這手牌的勝率(logistic)

用法:
  單手(文字):python3 scripts/hand_score.py --cards "2S 2H AD KS KC 10H 9C 8D 7S 6H 5C 4D 3C"
                點數 3..10/T/J/Q/K/A/2;花色 C梅花 D方塊 H紅心 S黑桃
  單手(id): python3 scripts/hand_score.py --ids 1,5,9,13,17,21,25,29,33,37,41,45,49
  整批 jsonl: python3 scripts/hand_score.py --jsonl ppo/data/online_games.jsonl --seats humans
                --seats humans|ours|all;逐手輸出 jsonl 到 stdout,結尾印彙總
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.hand_features_v2 import V2_FEATURES, v2_features

_RANKS = {"3": 0, "4": 1, "5": 2, "6": 3, "7": 4, "8": 5, "9": 6, "10": 7, "T": 7,
          "J": 8, "Q": 9, "K": 10, "A": 11, "2": 12}
_SUITS = {"C": 0, "D": 1, "H": 2, "S": 3}


def parse_cards(text):
    ids = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip().upper()
        r, s = tok[:-1], tok[-1]
        if r not in _RANKS or s not in _SUITS:
            raise ValueError(f"看不懂的牌:{tok}(例:2S、10H、QD)")
        ids.append(_RANKS[r] * 4 + _SUITS[s] + 1)
    return ids


class HandScorer:
    def __init__(self, weights_path=None):
        weights_path = weights_path or os.path.join(
            os.path.dirname(__file__), "..", "reports", "hand_score_v2_weights.json")
        with open(weights_path) as f:
            wj = json.load(f)
        self.keys = wj["features"]
        assert self.keys == [k for k, _ in V2_FEATURES], "權重檔與特徵表順序不符"
        self.mu = wj["standardize"]["mean"]
        self.sd = wj["standardize"]["std"]
        self.ridge_w = wj["ridge_score"]["coef"]
        self.ridge_b = wj["ridge_score"]["intercept"]
        self.logit_w = wj["logistic_win"]["coef"]
        self.logit_b = wj["logistic_win"]["intercept"]

    def score(self, cards):
        if len(cards) != 13 or len(set(cards)) != 13 or not all(1 <= c <= 52 for c in cards):
            raise ValueError("需要 13 張互不重複、id 1..52 的牌")
        f = v2_features(cards)
        z = [(f[k] - m) / s for k, m, s in zip(self.keys, self.mu, self.sd)]
        exp_score = sum(w * v for w, v in zip(self.ridge_w, z)) + self.ridge_b
        lin = sum(w * v for w, v in zip(self.logit_w, z)) + self.logit_b
        return {"exp_score": exp_score, "win_prob": 1 / (1 + math.exp(-lin))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", help='文字牌,例 "2S AH 10D ..."(13 張)')
    ap.add_argument("--ids", help="逗號分隔 card id,例 1,5,9,...(13 張)")
    ap.add_argument("--jsonl", help="online_games 格式 jsonl,逐手評分")
    ap.add_argument("--seats", default="humans", choices=["humans", "ours", "all"])
    ap.add_argument("--weights", default=None)
    args = ap.parse_args()

    scorer = HandScorer(args.weights)

    if args.cards or args.ids:
        cards = parse_cards(args.cards) if args.cards else [int(x) for x in args.ids.split(",")]
        r = scorer.score(cards)
        print(f"exp_score = {r['exp_score']:+.2f}(真人期望得分)")
        print(f"win_prob  = {r['win_prob']*100:.1f}%(真人勝率;公平基準 25%)")
        return

    if args.jsonl:
        n, tot_s, tot_p = 0, 0.0, 0.0
        for line in open(args.jsonl):
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            ours = int(g["our_seat"])
            for seat in range(4):
                if args.seats == "humans" and seat == ours:
                    continue
                if args.seats == "ours" and seat != ours:
                    continue
                r = scorer.score(g["hands"][str(seat)])
                print(json.dumps({"id": g.get("id"), "run": g.get("run"), "seat": seat,
                                  "is_ours": seat == ours,
                                  "exp_score": round(r["exp_score"], 3),
                                  "win_prob": round(r["win_prob"], 4)}))
                n += 1
                tot_s += r["exp_score"]
                tot_p += r["win_prob"]
        print(f"[summary] hands={n} mean_exp_score={tot_s/n:+.3f} mean_win_prob={tot_p/n:.4f}",
              file=sys.stderr)
        return

    ap.error("要給 --cards / --ids / --jsonl 其中一個")


if __name__ == "__main__":
    main()
