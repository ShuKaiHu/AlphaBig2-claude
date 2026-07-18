# Plan: 純 policy 引入分數訊號(2026-07-17)

範圍限定:**只動 policy,不碰 MCTS/value/belief**(使用者決定:search 會讓歸因複雜化)。
線上 control arm = `ab1_policyonly_p4500`(300 局,進行中),未來新 policy 上線就跟它同款設定對打。

## 0. 北極星與病灶(已驗證的事實)

北極星:線上對真人 avg_score(不是 win rate)。總缺口 −7.7/局,其中勝率通道占 77%。
policy_4500 = 純 BC(train_bc.py:73 無加權 cross_entropy),從未見過分數訊號。已定位病灶:

| 病灶 | 乾淨基準(bomb-free 層) | 真人錨點 | 價值 |
|---|---|---|---|
| G1 強牌轉換 | mp≤6 勝率 19.3%;領出五張 32.1%;起手五張完整打出 40.8% | 42.8% / 47.4% / 59.7% | ~+1.9/局 |
| G2 輸勢囤2 | P(留2\|被關輸局,發到2) = 70.8% | 57.4% | +0.4~0.5/局 |
| G3 炸彈壓制意願(post-fix 才可測) | 2/12 機會局 | 68.8% | 監控,價值未定 |
| G4 不退步約束 | humanness / vs-pool avg / 線上 avg | 不劣化 | 保險 |

資料使用規則:輸局側指標歷史照用(污染≤2.5%);強牌側一律 bomb-free 層;炸彈使用指標只認 post-fix。

## 1. 訓練方法候選(排序後逐一,不並行)

| # | 方法 | 一句話 | 為什麼是這個順序 |
|---|---|---|---|
| M1 | **結果加權 BC(AW-BC)** | 照樣只模仿真人,但「打出超額成績的真人」權重高 | 支撐約束=不可能發明非人類招 → 無漂移風險;改動~30行;直接對argmax坍縮機制 |
| M2 | 狀態條件加權(M1 的加強版) | 對輸勢/持2/強牌狀態把權重對比拉大 | 只在 M1 gate 部分通過時啟用 |
| M3 | BC 起點 + PPO 微調(KL 錨定 + pool 對手) | 真的 RL,平台計分當 reward | RL1/RL2 雙 null 前科;但當時沒有機制 gate。M1 結論出來前不碰 |
| M4 | 搜尋蒸餾 | — | **本輪排除**(使用者範圍限定;且 value tanh/13 飽和讓蒸餾目標不可信) |

### M1 配方
1. 運氣基準:`baseline(手) = E[真人分數 | min_plays, 2張數, 有無炸彈]`(全體真人分箱/回歸)
2. 每局每家 `advantage = 實際分 − baseline(起手牌)` → 該家的每個決策共權重
   `w = clip(exp(advantage/T), lo, hi)`,批內正規化均值=1
3. `T` 調到 ESS ≥ 50%(資料 ~6,900 局,policy 均勻餵食在 1,500 局已飽和 → 有 4 倍餘裕換質)
4. loss 一行:`(w * F.cross_entropy(logits, pos, reduction="none")).mean()`
5. checkpoint 選擇:vs-pool(standing rule),不用 val_acc

已知極限:credit assignment 是局級的(整局共享權重)。夠不夠由 gate 說話,不預先加複雜度。

## 2. 驗證階梯(便宜→貴,每個方法都走同一座梯子)

| 層 | 內容 | 成本 | 過/不過 |
|---|---|---|---|
| V0 訓練健檢 | 權重分布(ESS、clip率)、val top1 不崩(比 77.4% 掉 <5pp) | 分鐘 | 崩=回調 T |
| V1 **離線機制 gate**(主戰場) | self-play vs 固定 pool 各數千局,量 G1/G2 指標 + humanness + vs-pool avg | 小時(純policy無搜尋,快) | 見下方 gate 表 |
| V2 配對固定牌局 | paired_eval 消發牌運氣,新舊 policy 配對差 | 小時 | 輔助,不單獨裁決 |
| V3 線上 A/B | 新 policy vs policy_4500,同純 policy 模式、交錯、300-700/臂 | 1-2天 | 主指標=機制指標;avg non-inferiority |

### V1 gate(事前註冊,離線相對值:跟 policy_4500 的離線基線比,方向朝真人錨點)
- G2 囤2率:顯著下降(需 ~500 個被關事件/臂 ≈ 5-6k 局 self-play)
- G1 領出五張率 & 完整打出率:顯著上升(強牌局 ~11%,數千局足夠)
- G4 humanness(measure_pool_humanness.py)不降、vs-pool avg 不降
- 全過 → V3;部分過 → M2;全不過 → 記 null,考慮 M3

⚠️ 離線絕對值會跟線上不同(對手不是真人)——所以 V1 先建 policy_4500 的**離線基線卡**,一切比較同框。

## 3. 問題佇列(一個一個來)

- [ ] P0 凍結評估工具:`eval_policy_gates.py`(輸入 checkpoint → 輸出全部 gate 指標)
- [ ] P1 policy_4500 離線基線卡(gate 指標的離線絕對值)
- [ ] P2 權重管線 + 健檢報告(先看權重長相,再訓練)
- [ ] P3 訓練 AW-BC v1(局級權重,T 掃 2-3 個值)
- [ ] P4 V1 gate 評估 → 決定 M2 或前進
- [ ] P5 線上 A/B(等 ab1 跑完騰出帳號;control 資料已在手)
- [ ] P6 依結果:M2 / M3 / 收工寫結論

平行的既有工作(不衝突):ab1 線上戰役進行中;918 局收割在另一個 session 跑;value 去飽和留給未來 search 線。
