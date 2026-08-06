#!/usr/bin/env python3
"""手牌評分 v2:多指標加權指數(權重從真人資料學)。

════════ 事前註冊(v2,寫死後跑)════════
- 有效資料:真人三家(排除我們的座位;agent 轉換差會污染「牌→結果」)。
- 評估:5-fold 交叉驗證,**按局分組**(同局三家真人進同一 fold,防洩漏),
  一律報 out-of-fold 指標:score 用 R²,win 用 AUC 與 point-biserial R²。
- 正式 v2 評分 = 標準化指標的線性加權(可解釋):
    v2_score(win)  = logistic(win ~ features) 的線性部;
    v2_score(score)= ridge(score ~ features)。
  GBM(HistGradientBoosting)只作「這組指標的資訊上限」探針,不當正式評分。
- v1(等權 8 項)在同一 folds 上重算作基線。
- 不回頭改 v1 公平性判讀(該檢定與加權無關)。
═══════════════════════════════════════════

用法:python3 scripts/hand_score_v2.py
輸出:stdout + reports/hand_score_v2.md + reports/hand_score_v2_weights.json
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.hand_features_v2 import V2_FEATURES, v2_features
from scripts.hand_luck_report import COMPOSITE_KEYS, INDEX_DEFS, hand_indices, load_games

SEED = 42
N_FOLDS = 5


def main():
    root = os.path.join(os.path.dirname(__file__), "..")
    games, _ = load_games(os.path.join(root, "ppo", "data", "online_games.jsonl"))
    G = len(games)
    fkeys = [k for k, _ in V2_FEATURES]

    rows, win, score, game_id = [], [], [], []
    v1keys = [k for k, _, _, _ in INDEX_DEFS]
    v1sign = {k: s for k, _, s, _ in INDEX_DEFS}
    v1rows = []
    for gi, g in enumerate(games):
        ours = int(g["our_seat"])
        for seat in range(4):
            if seat == ours:
                continue
            cards = g["hands"][str(seat)]
            f = v2_features(cards)
            rows.append([f[k] for k in fkeys])
            v1 = hand_indices(cards)
            v1rows.append([v1[k] for k in v1keys])
            win.append(1.0 if int(g["winner"]) == seat else 0.0)
            score.append(float(g["scores"][str(seat)]))
            game_id.append(gi)
    X = np.array(rows)
    v1X = np.array(v1rows)
    y_win = np.array(win)
    y_sc = np.array(score)
    game_id = np.array(game_id)
    n = len(X)

    # v1 等權綜合(同資料標準化;等權不需 fit,無過擬合問題)
    v1z = np.stack([v1sign[k] * (v1X[:, i] - v1X[:, i].mean()) / v1X[:, i].std()
                    for i, k in enumerate(v1keys) if k in COMPOSITE_KEYS], axis=1).mean(1)

    # ── 5-fold 按局分組 CV ──
    rng = np.random.default_rng(SEED)
    fold_of_game = rng.integers(0, N_FOLDS, G)
    fold = fold_of_game[game_id]

    def zfit(tr_X, X_):
        mu, sd = tr_X.mean(0), tr_X.std(0)
        sd[sd == 0] = 1
        return (X_ - mu) / sd

    oof = {m: np.empty(n) for m in ["ridge_sc", "logit_win", "gbm_sc", "gbm_win"]}
    for f in range(N_FOLDS):
        tr, te = fold != f, fold == f
        Ztr, Zte = zfit(X[tr], X[tr]), zfit(X[tr], X[te])
        oof["ridge_sc"][te] = Ridge(alpha=1.0).fit(Ztr, y_sc[tr]).predict(Zte)
        oof["logit_win"][te] = LogisticRegression(C=1.0, max_iter=2000).fit(Ztr, y_win[tr]).predict_proba(Zte)[:, 1]
        oof["gbm_sc"][te] = HistGradientBoostingRegressor(random_state=SEED).fit(X[tr], y_sc[tr]).predict(X[te])
        oof["gbm_win"][te] = HistGradientBoostingClassifier(random_state=SEED).fit(X[tr], y_win[tr]).predict_proba(X[te])[:, 1]

    def r2_pred(pred, y):
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    def pb_r2(s, y):
        c = np.corrcoef(s, y)[0, 1]
        return c * c

    res = {
        "v1 等權(基線)":    {"R2_sc": pb_r2(v1z, y_sc), "R2_win": pb_r2(v1z, y_win), "AUC": roc_auc_score(y_win, v1z)},
        "v2 ridge(score)":  {"R2_sc": r2_pred(oof["ridge_sc"], y_sc), "R2_win": pb_r2(oof["ridge_sc"], y_win), "AUC": roc_auc_score(y_win, oof["ridge_sc"])},
        "v2 logistic(win)": {"R2_sc": pb_r2(oof["logit_win"], y_sc), "R2_win": pb_r2(oof["logit_win"], y_win), "AUC": roc_auc_score(y_win, oof["logit_win"])},
        "GBM 上限探針":      {"R2_sc": r2_pred(oof["gbm_sc"], y_sc), "R2_win": pb_r2(oof["gbm_win"], y_win), "AUC": roc_auc_score(y_win, oof["gbm_win"])},
    }

    # 二元雜訊天花板:若 logistic 機率已校準,win 的可解釋上限 = Var(p)/Var(y)
    p = oof["logit_win"]
    ceil_win = p.var() / y_win.var()

    # ── 全資料 fit 最終權重(部署用)──
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    ridge = Ridge(alpha=1.0).fit(Z, y_sc)
    logit = LogisticRegression(C=1.0, max_iter=2000).fit(Z, y_win)
    weights = {
        "features": fkeys,
        "standardize": {"mean": mu.tolist(), "std": sd.tolist()},
        "ridge_score": {"coef": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)},
        "logistic_win": {"coef": logit.coef_[0].tolist(), "intercept": float(logit.intercept_[0])},
        "fit_on": "human seats only, online_games.jsonl n=%d" % n,
    }
    wpath = os.path.join(root, "reports", "hand_score_v2_weights.json")
    os.makedirs(os.path.dirname(wpath), exist_ok=True)
    with open(wpath, "w") as fh:
        json.dump(weights, fh, indent=1)

    lines = []
    w = lines.append
    w("# 手牌評分 v2:加權指數(權重學自真人資料)")
    w("")
    w(f"- 有效資料:真人三家 n={n:,}(排除我們);評估 = 5-fold **按局分組** CV,全部 out-of-fold。")
    w(f"- v2 指標 {len(fkeys)} 項;權重檔:`reports/hand_score_v2_weights.json`。")
    w("")
    w("## 模型比較(out-of-fold)")
    w("")
    w("| 評分 | R²(score) | R²(win) | AUC(win) |")
    w("|---|---|---|---|")
    for name, r in res.items():
        w(f"| {name} | {r['R2_sc']:.3f} | {r['R2_win']:.3f} | {r['AUC']:.3f} |")
    w("")
    w(f"win 的二元雜訊天花板估計(校準機率變異/總變異):R²(win) ≲ **{ceil_win:.3f}**。")
    w("")
    w("## v2 權重(標準化係數,絕對值 = 重要度)")
    w("")
    w("| 指標 | 說明 | ridge(score) | logistic(win) |")
    w("|---|---|---|---|")
    order = np.argsort(-np.abs(ridge.coef_))
    for i in order:
        k, desc = V2_FEATURES[i]
        w(f"| {k} | {desc} | {ridge.coef_[i]:+.2f} | {logit.coef_[0][i]:+.2f} |")
    w("")
    w("## 用法")
    w("")
    w("v2 分數 = Σ 標準化指標 × 權重(+截距)。ridge 版直接讀作「這手牌的期望 score」,")
    w("logistic 版經 sigmoid 讀作「真人拿這手牌的勝率」。")

    report = "\n".join(lines) + "\n"
    os.makedirs(os.path.join(root, "reports"), exist_ok=True)
    out = os.path.join(root, "reports", "hand_score_v2.md")
    with open(out, "w") as fh:
        fh.write(report)
    print(report)
    print(f"[written] {out}\n[written] {wpath}", file=sys.stderr)


if __name__ == "__main__":
    main()
