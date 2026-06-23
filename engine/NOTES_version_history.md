# AlphaBig2 歷代版本 + 對人對戰資料(整理於 2026-06-23)

> 勝利指標 = **reward(avg_score)**,不是 win_rate。最終只認**線上對真人**資料(離線只是篩選器)。

---

## 1. 歷代版本內容

| 版本 | 核心改動 | 關鍵離線指標 | 線上對真人 | 結論 / 檔案 |
|------|----------|--------------|------------|-------------|
| **V6** | dominance 特徵(4維「我這張會不會被壓」),STATIC_DIM 306 | — | -2.26(**bug 時代,無效**) | saved/v6_dominance_deploy.pt |
| **V7** | value-coef=3 放大 value 權重 | 放大噪音、收斂變差 | 未測 | **失敗棄用**(checkpoint 已遺失) |
| **V8** | TD(λ) value targets(forward-view,λ=0.9,以 MCTS root value bootstrap) | value corr 早期 0.13 | bug 時代 +1.80(後證實是假);乾淨 ~-4.83 | saved/v8_td_deploy.pt |
| **V9a** | **全資訊(上帝視角)value net** 做 MCTS 葉評估 + 4 世界 determinization | value corr 0.13→0.25、跟牌浪費 34%→24%、MCTS-80 +5.64 | -4.45(64局);+v2 採樣合併 130局 **-3.80** | **現役 best.pt** = saved/v9a_fullinfo_deploy.pt |
| **V9b** | + 灰階 combo 特徵(單/對/順/葫/鐵/同花順 各一強度) | value 盲目診斷 47%(≈擲銅板,沒學起來) | 未乾淨測 | **特徵≠會用**;saved/v9b_combo_deploy.pt |
| **V9c** | combo 特徵 + **強對手 self-play**(strong-opp 0.5,smart heuristic) | 盲目診斷 44→**57%**、可避免拆組合 41→**29%**、MCTS-80 +6.93 | 環境變數搞混,從沒乾淨測到 | saved/v9c_combo_strongopp_deploy.pt |
| **引擎修正** | **跟牌允許主動 PASS**(returnAvailableActions 本來漏了 → V6–V9c 全都被迫每手出牌、拆組合) | 熱修 V9c 跟牌 pass 0→60%、拆組合 64→42% | — | big2Game.py(2026-06-17) |
| **V9d** | V9c + 訓練時開 voluntary pass | **塌縮**:policy P(pass)=0.9%、盲目診斷 55% | 未乾淨測 | 能力≠會用;saved/v9d_voluntarypass_deploy.pt |
| **V9e** | + **combo-aware heuristic**(BC+強對手會留組合、會 pass 小牌) | P(pass) 0.9→8.3%(9倍但仍低)、盲目診斷 46% | 未測 | 訊號仍太弱;saved/v9e_comboaware_deploy.pt |

**貫穿全系列的教訓(V7/V9b/V9d/V9e 四次證實)**:加「特徵/能力」≠ 模型會用,缺的是「訓練訊號」。
對 heuristic 對手,拆組合/亂送大牌不會被懲罰(贏定了)→ 學不會珍惜組合。**heuristic self-play
本質教不出像人的留組合打法。** 真正槓桿 = 更強/像人的訓練訊號(reward shaping 或真人棋譜)。

---

## 2. 對人對戰資料(線上)

### 可靠的結論
- **🔴 鐵律#1:離線 ≠ 線上。** 離線對 heuristic V6>V8,線上對真人卻 V8>V6。對 heuristic 強 ≠ 對真人強。
- **🔴 鐵律#2:bug 時代所有數字無效。** 2026-06-12 之前有三個 executor bug(座位旋轉、control 誤判、
  一張牌規則),早期看到的正分(06-02 +0.6、06-04 +0.7、06-10 +0.5、V8「+1.80 贏真人」)**全是假象**。
- **乾淨基準(修完 executor 後):所有最強模型對真人 ≈ 每局 -4 ~ -5**,沒有任何一個明顯突破。
  - V8 乾淨 ≈ -4.83 ｜ V9a -4.45(64局)｜ V9a+v2 合併 130局 **-3.80**(目前最可靠的最佳值)。
- **2026-06-16「V9c」那批(-6.72/-7.59)其實是 V9a**(使用者忘設 ALPHA_BIG2_CKPT → 跑預設 best.pt),
  外加 3 局誤用 codex transformer。**真 V9c/V9d/V9e 至今都沒有乾淨的線上對真人資料。**
- **2026-06-23 V9c+passfix:災難性過度 pass**(連兩家剩 1 張還在留牌 → 滿手 13 張 -52),
  policy pass 偏執 0.8~0.98。盲目開放 pass 不可行。

### 原始每日紀錄(artifacts/reward_log.jsonl;★=可靠歸因,其餘多為 bug 時代或模型未知)
| 日期 | 局數 | avg | 中位 | 最差 | 備註 |
|------|------|-----|------|------|------|
| 06-02 | 73 | +0.60 | -3 | -36 | bug 時代,無效 |
| 06-04 | 79 | +0.70 | -2 | -80 | bug 時代,無效 |
| 06-10 | 149 | +0.50 | -3 | -28 | bug 時代(control bug 當「出最小牌」拐杖) |
| 06-11 | 101 | -5.17 | -4 | -44 | V9a 首批(修正版 parser) |
| 06-12 | 55 | -8.29 | -6 | -176 | V8 乾淨 baseline(剔除 -176 後 ~-4.83) |
| 06-14 | 69 | -1.13 | -4 | -40 | V9a+v2(好手氣 session,含 +79) |
| 06-15 | 61 | -6.82 | -5 | -24 | V9a+v2(與上批合併 = -3.80) |
| 06-16 | 102 | -7.59 | -5 | -88 | ★ 實為 V9a(忘設env)+ 3局codex |
| 06-17~06-22 | 多批 | -5 ~ -10 | — | — | 模型歸因不明(多半非本專案 wrapper / 無 ckpt 標記) |
| 06-23 | (進行中) | -52×n | -52 | -52 | ★ V9c+passfix,過度 pass 災難 |

> 注:2026-06-16 起 mcts_moves.jsonl 每筆有 `ckpt` 欄位可精準歸因;之前只能靠日期+記憶推斷。

### 戰略現況(2026-06-23)
1. 「給模型更好的資訊/特徵/採樣」這條軸(全資訊 value、v2 採樣、combo 特徵)**離線都進步、線上就是不動**。
2. 「讓模型會 pass / 留組合」用 heuristic 教,四輪(V9b–V9e)都失敗(value 仍對組合盲目)。
3. 對真人 -4/局是真實實力差距。**下一步槓桿**:reward shaping(用 combo_strength 在 value target
   直接罰拆高強度組合 + 對手快出完時別留牌)讓 pass 變「有選擇性」;或真人棋譜模仿。
