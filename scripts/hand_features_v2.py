"""v2 手牌特徵(19 項)— 訓練(hand_score_v2.py)與推論(hand_score.py)共用。

炸彈/順子窗一律循 value/hand_features.py 正典;不要在別處重抄這份特徵表。
卡牌編碼:card id 1..52;rank_idx=(c-1)//4(0='3'…12='2');suit=(c-1)%4(0=梅花 1=方塊 2=紅心 3=黑桃)。
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.hand_luck_report import hand_indices
from value.hand_features import STRAIGHT_WINDOWS_RIDX

# (名稱, 說明)——順序即權重檔 features 欄位順序
V2_FEATURES = [
    ("cnt_2",       "2 的張數"),
    ("cnt_A",       "A 的張數"),
    ("cnt_K",       "K 的張數"),
    ("cnt_Q",       "Q 的張數"),
    ("cnt_J",       "J 的張數"),
    ("spade2",      "持有黑桃 2"),
    ("top1_card",   "最大單張(card id /52)"),
    ("top3_mean",   "前三大單張平均(card id /52)"),
    ("sum_rank",    "點數總和"),
    ("sum_top5",    "前五大點數和"),
    ("n_pairs",     "可拆對子數"),
    ("n_trips",     "三條點數數"),
    ("n_quads",     "鐵支次數"),
    ("n_sf",        "同花順次數"),
    ("n_straightw", "順子潛力(點數覆蓋的 10 窗數)"),
    ("max_suit",    "最長花色張數"),
    ("min_plays",   "greedy 最少出完手數"),
    ("n_dead_low",  "低廢單張數(3-6 孤張)"),
    ("n_low7",      "低張數(3-7)"),
]


def v2_features(cards):
    ridx = [(c - 1) // 4 for c in cards]
    rc = Counter(ridx)
    base = hand_indices(cards)          # 重用 v1 正典計算(炸彈/對子/min_plays…)
    ids = sorted(cards, reverse=True)
    rset = set(ridx)
    suit_cnt = Counter((c - 1) % 4 for c in cards)
    return {
        "cnt_2": rc.get(12, 0),
        "cnt_A": rc.get(11, 0),
        "cnt_K": rc.get(10, 0),
        "cnt_Q": rc.get(9, 0),
        "cnt_J": rc.get(8, 0),
        "spade2": base["spade2"],
        "top1_card": ids[0] / 52.0,
        "top3_mean": sum(ids[:3]) / 3 / 52.0,
        "sum_rank": base["sum_rank"],
        "sum_top5": sum(sorted(ridx)[-5:]),
        "n_pairs": base["n_pairs"],
        "n_trips": base["n_trips"],
        "n_quads": base["n_quads"],
        "n_sf": base["n_sf"],
        "n_straightw": sum(1 for w in STRAIGHT_WINDOWS_RIDX if w <= rset),
        "max_suit": max(suit_cnt.values()),
        "min_plays": base["min_plays"],
        "n_dead_low": base["n_dead_low"],
        "n_low7": sum(1 for r in ridx if r <= 4),
    }
