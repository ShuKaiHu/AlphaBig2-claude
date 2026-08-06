#!/usr/bin/env python3
"""牌運統計:神來也發牌對我們公不公平?

資料:ppo/data/online_games.jsonl(每局四家起手 13 張 + our_seat)。

════════ 事前判讀規則(寫死,看到結果後不許改)════════
主檢定(是否針對我們):綜合指標的組內置換檢定(同一局四家座位可交換)。
  - 「被針對(拿爛牌)」成立:雙尾 p < 0.01 且方向為負(我們 < 同桌平均)。
  - 「被眷顧」成立:雙尾 p < 0.01 且方向為正。
  - 其餘一律記 NULL(牌運正常,只是感覺)。
副檢定(洗牌器整體偏差):我們的手牌 vs 均勻發牌 MC null(N=100,000, seed=42),
  逐指標 z 檢定。此檢定顯著只說明洗牌器整體不均,不能推論「針對我們」。
逐項指標:Holm 校正 α=0.05,僅作輔助定位,不單獨改變主判讀。
排除資料只按機制:非 4 家 × 13 張、或 52 張不成完整分割的局;不按結果排除。
綜合指標(事前定義,等權):下列 8 項對 MC null 標準化後取平均,方向對齊「高=好」:
  +n_twos +n_aces +n_high +sum_rank +has_bomb +n_pairs −min_plays −n_dead_low
═══════════════════════════════════════════════════════

卡牌編碼:card 1..52;rank_idx=(c-1)//4(0='3'…12='2');suit=(c-1)%4。
炸彈偵測一律用 value/hand_features.py 正典(鐵支/同花順;10 窗順子)。

用法:python3 scripts/hand_luck_report.py [--games ppo/data/online_games.jsonl]
輸出:stdout 摘要 + reports/hand_luck_report.md
"""
import argparse
import json
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from value.hand_features import has_bomb_rank_suit, min_plays_to_empty  # 正典,不許手寫

N_MC = 100_000
N_PERM = 100_000
SEED = 42

# ── 指標定義(名稱, 說明, 方向:+1 高=好 / -1 高=壞, 是否進綜合)──
INDEX_DEFS = [
    ("n_twos",     "2 的張數(最大點數)",              +1, True),
    ("n_aces",     "A 的張數",                          +1, True),
    ("n_high",     "高張數(K/A/2)",                    +1, True),
    ("sum_rank",   "點數總和(rank_idx 0-12 加總)",     +1, True),
    ("has_bomb",   "炸彈(鐵支/同花順,正典偵測)",     +1, True),
    ("n_pairs",    "可拆對子數(Σ 同點張數//2)",        +1, True),
    ("n_trips",    "三條以上點數數",                    +1, False),
    ("min_plays",  "greedy 最少出完手數(小=好)",       -1, True),
    ("n_dead_low", "低廢單張數(3-6 的孤張)",           -1, True),
    ("spade2",     "持有黑桃 2",                        +1, False),
]
COMPOSITE_KEYS = [k for k, _, _, inc in INDEX_DEFS if inc]


def hand_indices(cards):
    """13 張 card id → 指標 dict。"""
    ridx = [(c - 1) // 4 for c in cards]
    pairs = [((c - 1) // 4, (c - 1) % 4) for c in cards]
    rc = Counter(ridx)
    return {
        "n_twos": rc.get(12, 0),
        "n_aces": rc.get(11, 0),
        "n_high": sum(rc.get(r, 0) for r in (10, 11, 12)),
        "sum_rank": sum(ridx),
        "has_bomb": int(has_bomb_rank_suit(pairs)),
        "n_pairs": sum(n // 2 for n in rc.values()),
        "n_trips": sum(1 for n in rc.values() if n >= 3),
        "min_plays": min_plays_to_empty(cards),
        "n_dead_low": sum(1 for r, n in rc.items() if r <= 3 and n == 1),
        "spade2": int(52 in cards),
    }


def load_games(path):
    """回傳 (games, 排除計數 dict)。排除只按機制。"""
    games, excl = [], Counter()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            g = json.loads(line)
            hands = g.get("hands", {})
            if g.get("our_seat") is None or str(g["our_seat"]) not in hands:
                excl["缺 our_seat"] += 1
                continue
            if len(hands) != 4 or any(len(h) != 13 for h in hands.values()):
                excl["非 4 家 × 13 張"] += 1
                continue
            allc = sorted(c for h in hands.values() for c in h)
            if allc != list(range(1, 53)):
                excl["52 張不成完整分割"] += 1
                continue
            games.append(g)
    return games, excl


def norm_p_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def holm(pvals):
    """Holm-Bonferroni 校正後 p(保持單調)。"""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=os.path.join(os.path.dirname(__file__), "..", "ppo", "data", "online_games.jsonl"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "hand_luck_report.md"))
    args = ap.parse_args()

    games, excl = load_games(args.games)
    G = len(games)
    keys = [k for k, _, _, _ in INDEX_DEFS]
    sign = {k: s for k, _, s, _ in INDEX_DEFS}

    # ── 全部 4G 手牌的指標矩陣:vals[k] shape (G, 4),欄 = 座位 0..3 ──
    vals = {k: np.empty((G, 4)) for k in keys}
    our_col = np.empty(G, dtype=int)
    for gi, g in enumerate(games):
        our_col[gi] = int(g["our_seat"])
        for seat in range(4):
            idx = hand_indices(g["hands"][str(seat)])
            for k in keys:
                vals[k][gi, seat] = idx[k]

    # ── MC null:均勻發 13 張(我們那一手的邊際分佈)──
    rng = np.random.default_rng(SEED)
    deck = np.arange(1, 53)
    mc = {k: np.empty(N_MC) for k in keys}
    for i in range(N_MC):
        hand = rng.choice(deck, size=13, replace=False)
        idx = hand_indices(hand.tolist())
        for k in keys:
            mc[k][i] = idx[k]
    null_mu = {k: mc[k].mean() for k in keys}
    null_sd = {k: mc[k].std(ddof=1) for k in keys}

    # ── 綜合指標(事前定義):對 null 標準化、方向對齊、等權平均 ──
    def composite(mat_by_key):
        zs = [sign[k] * (mat_by_key[k] - null_mu[k]) / null_sd[k] for k in COMPOSITE_KEYS]
        return sum(zs) / len(zs)

    comp_vals = composite({k: vals[k] for k in COMPOSITE_KEYS})          # (G,4)
    comp_mc = composite({k: mc[k] for k in COMPOSITE_KEYS})              # (N_MC,)

    # ── 主檢定:組內置換(座位可交換)──
    def perm_test(mat):
        """mat: (G,4)。回傳 (T_obs, p, per-game centered our 值)。"""
        centered = mat - mat.mean(axis=1, keepdims=True)
        ours_c = centered[np.arange(G), our_col]
        T_obs = ours_c.mean()
        prng = np.random.default_rng(SEED + 1)
        picks = prng.integers(0, 4, size=(N_PERM, G))
        T_perm = centered[np.arange(G), picks].mean(axis=1)
        p = (1 + np.sum(np.abs(T_perm) >= abs(T_obs))) / (1 + N_PERM)
        return T_obs, p, ours_c

    comp_T, comp_p, _ = perm_test(comp_vals)
    perm_rows = {}
    for k in keys:
        T, p, _ = perm_test(vals[k])
        perm_rows[k] = (T, p)

    # ── 副檢定:我們的手 vs MC null(跨局獨立)──
    ours = {k: vals[k][np.arange(G), our_col] for k in keys}
    abs_rows = {}
    for k in keys:
        se = null_sd[k] / math.sqrt(G)
        z = (ours[k].mean() - null_mu[k]) / se
        abs_rows[k] = (z, norm_p_two_sided(z))
    comp_ours = comp_vals[np.arange(G), our_col]
    comp_se = comp_mc.std(ddof=1) / math.sqrt(G)
    comp_abs_z = (comp_ours.mean() - comp_mc.mean()) / comp_se
    comp_abs_p = norm_p_two_sided(comp_abs_z)

    # Holm(逐項,置換 p;綜合為主檢定不入校正)
    holm_p = holm([perm_rows[k][1] for k in keys])
    holm_map = dict(zip(keys, holm_p))

    # ── 牌運分:每局我們的綜合指標在 MC null 的百分位 ──
    luck_pct = 100.0 * np.searchsorted(np.sort(comp_mc), comp_ours) / len(comp_mc)

    # ── 判讀(照檔頭事前規則)──
    if comp_p < 0.01 and comp_T < 0:
        verdict = "**被針對(拿爛牌)成立**:綜合指標顯著低於同桌"
    elif comp_p < 0.01 and comp_T > 0:
        verdict = "**被眷顧**:綜合指標顯著高於同桌"
    else:
        verdict = "**NULL:沒有證據顯示神來也發給我們的牌比同桌差**"

    opp_mask = np.ones((G, 4), dtype=bool)
    opp_mask[np.arange(G), our_col] = False

    lines = []
    w = lines.append
    w("# 牌運統計報告:神來也發牌公平性")
    w("")
    w(f"- 資料:`{os.path.relpath(args.games, os.path.join(os.path.dirname(__file__), '..'))}`,納入 **{G:,} 局**;排除 {dict(excl) if excl else '0 局'}(只按機制)。")
    w(f"- MC null:均勻發牌 {N_MC:,} 手(seed={SEED});置換 {N_PERM:,} 次。")
    w("- 事前判讀規則見 `scripts/hand_luck_report.py` 檔頭(先寫死後跑)。")
    w("")
    w("## 主判讀")
    w("")
    w(f"{verdict}")
    w("")
    w(f"- 綜合指標(z 單位):我們 − 同桌平均 = **{comp_T:+.4f}**,置換雙尾 p = **{comp_p:.4f}**")
    w(f"- 綜合指標 vs 均勻 null:z = {comp_abs_z:+.2f},p = {comp_abs_p:.4f}(洗牌器整體檢定,非針對性)")
    w(f"- 平均牌運分(綜合指標之 null 百分位,50 = 恰好中位):**{luck_pct.mean():.2f}**")
    w("")
    w("## 逐項指標")
    w("")
    w("| 指標 | 說明 | 方向 | 我們平均 | 同桌平均 | null 平均±SD | 置換 p | Holm p | vs null z |")
    w("|---|---|---|---|---|---|---|---|---|")
    for k, desc, s, inc in INDEX_DEFS:
        T, p = perm_rows[k]
        z, _ = abs_rows[k]
        opp_mean = vals[k][opp_mask].mean()
        star = " ✳" if holm_map[k] < 0.05 else ""
        w(f"| {k}{'†' if inc else ''} | {desc} | {'高=好' if s > 0 else '低=好'} | {ours[k].mean():.3f} | {opp_mean:.3f} | "
          f"{null_mu[k]:.3f}±{null_sd[k]:.3f} | {p:.4f}{star} | {holm_map[k]:.3f} | {z:+.2f} |")
    w("")
    w("† = 綜合指標成分。✳ = Holm 校正後 p<0.05。置換 p 回答「我們 vs 同桌」;vs null z 回答「洗牌器整體」。")
    w("")
    w("## 牌運分分佈(我們的每局綜合百分位)")
    w("")
    hist, _ = np.histogram(luck_pct, bins=10, range=(0, 100))
    for b in range(10):
        bar = "█" * int(round(60 * hist[b] / max(1, hist.max())))
        w(f"- {b*10:3d}–{b*10+10:3d}:{hist[b]:5d} {bar}")
    w("")
    w(f"最爛 1% 分位手牌出現率:{(luck_pct < 1).mean()*100:.2f}%(公平期望 1%);"
      f"最好 1%:{(luck_pct > 99).mean()*100:.2f}%。")
    w("")
    w("## 方法備註")
    w("")
    w("- 組內置換檢定不依賴洗牌器整體是否均勻:就算洗牌器有全域偏差,四家同受其害,")
    w("  只有「針對我們的座位」才會讓置換檢定顯著。")
    w("- 同一局四家手牌互相負相關(分割一副牌),因此跨家比較一律用組內置換,不做跨家獨立假設。")
    w("- 手牌資料含對手起手 13 張(收割時全知重建);若重建有誤會同時污染四家,不偏向任一座位。")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(report)
    print(f"[written] {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
