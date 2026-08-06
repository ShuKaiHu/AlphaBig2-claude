#!/usr/bin/env python3
"""牌運評分 × 對局結果:綜合手牌分數(hand_luck_report 的事前定義)對
win/loss 與 score(reward)的預測力驗證。

- 資料:ppo/data/online_games.jsonl(含 winner 與四家 scores)。
- 綜合分:沿用 scripts/hand_luck_report.py 的事前定義(8 項標準化等權),不改。
- 相關一律報 R²(附方向):win/loss 用 point-biserial²+AUC;score 用 Pearson²+Spearman²。
- 主母體 = 真人三家(排除我們的座位;我們的 agent 轉換差會污染「手牌→結果」估計)。
  我們的座位另列作對照。同局三家真人互相依賴,R² 作描述用。

用法:python3 scripts/hand_luck_outcome.py [--n-mc 50000]
輸出:stdout + reports/hand_luck_outcome.md
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.hand_luck_report import COMPOSITE_KEYS, INDEX_DEFS, SEED, hand_indices, load_games


def rankdata(x):
    """平均秩(處理同分),純 numpy。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(a, b):
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    return float((a * b).mean())


def auc(score, label):
    """P(score_win > score_loss),同分算 0.5。"""
    r = rankdata(score)
    n1 = int(label.sum())
    n0 = len(label) - n1
    return float((r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=os.path.join(os.path.dirname(__file__), "..", "ppo", "data", "online_games.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "hand_luck_outcome.md"))
    ap.add_argument("--n-mc", type=int, default=50_000, help="null 標準化/百分位用的 MC 手數")
    args = ap.parse_args()

    games, _ = load_games(args.games)
    G = len(games)
    keys = [k for k, _, _, _ in INDEX_DEFS]
    sign = {k: s for k, _, s, _ in INDEX_DEFS}

    vals = {k: np.empty((G, 4)) for k in keys}
    win = np.zeros((G, 4))
    score = np.empty((G, 4))
    our_col = np.empty(G, dtype=int)
    for gi, g in enumerate(games):
        our_col[gi] = int(g["our_seat"])
        win[gi, int(g["winner"])] = 1
        for seat in range(4):
            score[gi, seat] = float(g["scores"][str(seat)])
            idx = hand_indices(g["hands"][str(seat)])
            for k in keys:
                vals[k][gi, seat] = idx[k]

    rng = np.random.default_rng(SEED)
    deck = np.arange(1, 53)
    mc = {k: np.empty(args.n_mc) for k in keys}
    for i in range(args.n_mc):
        idx = hand_indices(rng.choice(deck, 13, replace=False).tolist())
        for k in keys:
            mc[k][i] = idx[k]
    mu = {k: mc[k].mean() for k in keys}
    sd = {k: mc[k].std(ddof=1) for k in keys}

    comp = sum(sign[k] * (vals[k] - mu[k]) / sd[k] for k in COMPOSITE_KEYS) / len(COMPOSITE_KEYS)
    comp_mc = np.sort(sum(sign[k] * (mc[k] - mu[k]) / sd[k] for k in COMPOSITE_KEYS) / len(COMPOSITE_KEYS))
    luck_pct = 100.0 * np.searchsorted(comp_mc, comp) / len(comp_mc)     # (G,4)

    ar = np.arange(G)
    opp_mask = np.ones((G, 4), dtype=bool)
    opp_mask[ar, our_col] = False
    pops = {
        "真人三家(有效資料,排除我們)":
            (comp[opp_mask], win[opp_mask], score[opp_mask], luck_pct[opp_mask]),
        "我們的座位(對照,agent 轉換污染)":
            (comp[ar, our_col], win[ar, our_col], score[ar, our_col], luck_pct[ar, our_col]),
    }

    lines = []
    w = lines.append
    w("# 牌運評分 × 對局結果(win/loss 與 score 相關)")
    w("")
    w(f"- 資料:{G:,} 局;綜合分沿用 `scripts/hand_luck_report.py` 事前定義(未改)。")
    w(f"- null 標準化 MC = {args.n_mc:,} 手(seed={SEED})。")
    w("")
    for name, (c, wn, sc, lp) in pops.items():
        n = len(c)
        r_win = pearson(c, wn)
        r_sc = pearson(c, sc)
        rho_sc = pearson(rankdata(c), rankdata(sc))
        w(f"## {name}(n={n:,},勝率 {wn.mean()*100:.1f}%)")
        w("")
        w(f"- 綜合分 × win/loss:R² = **{r_win**2:.3f}**(方向 {'+' if r_win > 0 else '-'},r={r_win:+.3f}),AUC = **{auc(c, wn):.3f}**")
        w(f"- 綜合分 × score:Pearson R² = **{r_sc**2:.3f}**(r={r_sc:+.3f}),Spearman R² = **{rho_sc**2:.3f}**(ρ={rho_sc:+.3f})")
        w("")
        w("| 牌運分十分位 | n | 勝率 | 平均 score |")
        w("|---|---|---|---|")
        for b in range(10):
            m = (lp >= b * 10) & (lp < b * 10 + 10) if b < 9 else (lp >= 90)
            w(f"| {b*10}–{b*10+10} | {int(m.sum()):,} | {wn[m].mean()*100:.1f}% | {sc[m].mean():+.2f} |")
        w("")
    w("## 單項指標 × 結果(真人三家,R²,方向對齊「高=好」後全為正相關)")
    w("")
    w("| 指標 | 說明 | R²(win) | R²(score) | Spearman R²(score) |")
    w("|---|---|---|---|---|")
    flat_w, flat_s = win[opp_mask], score[opp_mask]
    for k, desc, s, _ in INDEX_DEFS:
        v = (vals[k] * s)[opp_mask]
        w(f"| {k} | {desc} | {pearson(v, flat_w)**2:.3f} | {pearson(v, flat_s)**2:.3f} | "
          f"{pearson(rankdata(v), rankdata(flat_s))**2:.3f} |")
    w("")
    w("備註:同局三家真人互相依賴(win 至多一家、score 近零和),R² 作描述用。")
    w("score 為神來也記分(勝者=三家剩牌和,敗者=−自己剩牌)。")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
