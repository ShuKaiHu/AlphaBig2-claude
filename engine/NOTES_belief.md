# Belief model — 階段性準確度觀念 (重要)

## 核心觀念 (使用者提出, 2026-06)

Belief model（推斷對手手牌）的準確度**必須分早期/晚期看，不能整局平均**：

- **早期（剛發牌）**：沒人出過牌，資訊量幾乎為 0 → belief 理論上**不可能準**，
  接近亂猜是正常的，不是 bug。
- **晚期（大量牌已出）**：看得到每家分別出了什麼牌、pass 過什麼 →
  資訊量大 → belief **應該要很準**。

而且**晚期正是 determinization 最關鍵的時候**（殘局精算），所以晚期 belief
準不準，直接決定線上 MCTS 在關鍵時刻的強度。

## 評估方法（正確做法）

不要報整局平均 lift（會被早期稀釋）。要**分階段 bucket**：
以「還有多少張牌未知（= 對手手上的牌）」當階段指標：
- early: >27 unknown
- mid:   15–27 unknown
- late:  <15 unknown

指標：`precision@k`（belief 排名前 k 的牌命中對手真實持牌的比例，
k = 該對手真實手牌數），對照 `base = k/unknown`（亂猜），看 `lift = prec/base`。
- lift 1.0 = 無用；>1.3 = 明顯有預測力。

工具：`probe_belief.py`（已分階段）。

## 量測紀錄

**deploy_best_mcts.pt（belief weight=0.1，舊）：**
| phase | lift |
|-------|------|
| early | 1.01x |
| mid   | 1.04x |
| late  | 1.07x |

→ 趨勢正確（晚期較高）但幅度太小 = under-trained（0.1 權重被 policy/value 壓過）。

## 假設與下一步

**假設**：把 belief 權重提高（0.1→0.4）重訓後，**晚期 lift 應大幅上升**，
早期維持 ~1.0（早期本就無資訊，學不出來也合理）。

**若假設成立 → 實作 belief 引導 determinization：**
- 晚期：用 belief 機率分布採樣對手手牌（取代均勻隨機）
- 早期：直接用均勻隨機（belief 沒用，省算力）
- 對應修改：`alpha_big2_wrapper.py` 的 `_sample_opponent_hands`

**若晚期 lift 仍上不去 → 退路**：belief-independent 的「多重 determinization 平均」
（每步猜 N 副牌各跑 MCTS，平均）—— robust 但較貴。

## 實證結論 (2026-06, eval_determinization.py, 80 games, 等算力=80 sims, vs heuristic)

| config | avg_score | 1st | 4th |
|--------|-----------|-----|-----|
| 單一 deep (1×80) uniform | **+12.08** ± 3.76 | 37.5% | 4 |
| 多重 (4×20) uniform | +9.99 ± 2.57 | 41.2% | 3 |
| belief 引導 (4×20) | **+4.14** ± 2.32 | 18.8% | 3 |

**重大發現:belief 引導採樣（用目前 lift~1.1 的弱 belief）反而比均勻隨機「更差」。**
不準的 belief 會把 determinization 偏向錯誤世界、降低多樣性 → 有害。
→ belief 引導的前提是「belief 必須夠準」；在 late-lift 衝到 ~1.3+ 之前，不要用。

多重 determinization 對 heuristic 對手沒明顯優勢（對手出牌可預測，與其確切手牌弱相關）。
→ 目前線上「單一 determinization + 1秒 deep MCTS」已接近最優。

**決策:暫停 belief 引導投資（資料顯示目前有害）。聚焦已驗證的槓桿 = value-net 訓練
（部署強度 +1.6→+7.1）。belief 工具鏈保留，待 belief 準度提升或線上實測暴露
determinization 弱點時再重啟。**

**對手依賴性警告:以上皆對 heuristic 對手測得。真人會詐唬、其手牌更相關，
belief/多重 determinization 對真人「可能」較有價值 —— 留待線上實測驗證。**

## 訓練教訓:不要 re-warm LR (2026-06)

在已收斂的 value net(+7.1, iter 414)上「re-warm LR」(用 --iterations 900 把 cosine
位置拉回中段、LR 回升)是**破壞性的**:
- belief 0.4 + re-warm LR → 部署強度 +7.1 → +2.2 (iter 438)
- belief 0.1 + re-warm LR → 部署強度仍只 +3.4 (iter 548),134 iters 沒爬回

→ 元兇是 re-warm LR 本身。原始「乾淨一次性 cosine 退火(2e-4→floor)」產生的 +7.1
   才是最佳。事後拉高 LR 會把網路震離好的最優點,且難以恢復。
→ 要超越 +7.1,正確做法是「全新更長的乾淨 cosine 訓練」或「更多 sims 的乾淨訓練」,
   不是 re-warm。或者直接用線上實測取得真實訊號。

部署模型鎖定:engine/checkpoints/{best,latest,deploy_best_mcts}.pt = +7.1 (perfect-info
MCTS 80sims), 不完全資訊單一 deep determinization eval 達 +12。

## belief 的正確抽象層 (使用者洞察, 2026-06) — 重要重構方向

之前 belief head 猜「哪個對手持有哪張牌」是錯的抽象層 (實測 1.1x 無用)。
人類真正用的 belief 是:
1. 型別層級排除:「下家沒順子」「上家沒 A」「對家沒 >10 的順」
2. 牌力支配:「我這張/組現在是不是最大、有沒有人壓得了」
3. 理性行為推論:從對手 pass/出牌反推手牌結構

### Tier A — 確定性,不需神經網路 (先做, 最高 CP)
① 牌力支配 (nuts) 特徵:從 (我的手牌 + 已出牌) 精確算未見牌,判斷我的
   單張/對/順/葫蘆…是否為「當前無敵」。免費、精確。直接回答「會不會被壓」。
② pass 揭露的 void:理性假設下,某人 pass 掉某型某level → 他沒有更大的該型。
   逐手累積成 per-opponent 硬約束。
   例:開 34567 下家 pass → 下家 void 順子≤34567。

### Tier B — 機率性,需好資料 (後做)
理性推論 (如:出 22255 葫蘆 → 推斷無其他葫蘆,因理性會先出小的)。
神經 belief 可學,但須對「出牌有資訊量的對手」(真人/強 self-play) 訓練。
笨 heuristic self-play 學不到 → 解釋了 belief head 為何練不起來。

### 連動 MCTS 的正確方式:約束式 determinization
- 上次失敗的「belief 軟性加權採樣」→ 偏向錯誤世界 → 有害。
- 正確:用 Tier A 硬約束做 determinization,排除違反 void 的世界
  (已知下家沒順,就不發能成順的牌給他)。每個假設世界都合乎推論 → MCTS 有效。
- 這就是使用者說的「把可能性做些排除」。

### 實作順序建議
1. Tier A void-tracking + dominance 特徵 (確定性, 純邏輯)
2. 約束式 determinization (用 void 約束採樣, 取代均勻/軟加權)
3. dominance 特徵餵進 feature encoding (policy/value 可直接用)
4. (後) Tier B 神經 belief, 需對強對手/真人資料訓練

### 重要:這些主要對「真人對手」有價值
heuristic 對手出牌可預測、pass 資訊量低 → determinization/void 對它幫助小
(已實證)。但真人大量使用 void/dominance 推理 → 此方向專門針對真人,
只能靠線上實測驗證。

## 線上實測 + 強度診斷 (2026-06-02) — 戰略結論

### 線上對真人 (~68 場, 資料有瑕疵)
- card-score 可靠片段(開頭12場): avg +2.33, median -2, 4 場出完
- 真實籌碼 bankroll: 淨 -470 (打平), 但單 session 振幅 ±13000
- **結論: 對真人 ≈ 打平, 非壓制。** Big2 reward 高變異, 均值被罕見全壘打/災難場主宰。

### 過擬合假設被推翻
模型對 weak heuristic +1.3, 對 smart heuristic(更像人) +2.6 → **沒有過擬合到弱對手**。
真相更樸素: **模型就是「heuristic 等級」, 真人比簡單 heuristic 強。**
差距是模型本身強度, 不是過擬合。

### 強度天花板 = 真正瓶頸
- 部署 +7.1 (vs heuristic) 但對真人打平 → self-play 收斂在約 amateur 水準。
- 同配方(sims=50)跑更長(v3, 800 iters)大概率只marginal提升, 不會突破到「贏真人」。
- 要突破需要改配方, 不是跑更久:
  1. **更多 self-play sims** (50→150+): 更深搜索 → 更強學習目標 (AlphaZero 標準槓桿)。代價: 每 iter ~3x 慢。
  2. **incorporate 人類知識**: Tier A void/dominance 推理 (模型現在沒有)。
  3. **league**: 對過去版本+多樣對手訓練 (目前停用)。
  4. **真人 game logs 訓練** (imitation from strong humans)。
- 注意: 純 self-play 在這種大動作空間+不完全資訊+此網路大小, 可能本質上收斂在 amateur。
  突破需要更大算力(AlphaZero-scale)或人類知識注入。

### 已知良好成果 (別丟失)
- executor 零失敗 (生產級)
- 4-player MCTS 修正 (核心 bug 已除)
- deploy_best_mcts.pt = 競爭級 agent, 對真人打平, 對簡單 heuristic 壓制

### 測量教訓
- capture_scores.py 有 seq-dedup bug (跨 session seq 重置 → 凍結)。下次: 用 wrapper 直接寫
  per-game append-only reward log, 或複合鍵 (session,seq)。
- game_results.jsonl 的 placement/my_remaining 不可靠 (低估出完數)。只信 server round_result。

## 輸牌深度分析 (2026-06-02): 不是 bug, 是被輾壓

深入回推線上最差一場 (-10):
- self 起手 6666炸彈 + QQQ + 散牌(5,7,7,8,T,J) — 普通牌
- left 開 AAA葫蘆 → self 用 6666 炸彈壓 (正確!)
- 對手後續: KK, 999+TT葫蘆, 4444炸彈, 222 — 怪物牌
- self 的 PASS 幾乎全是被迫 (壓不過), 卡住的中間牌是被對手高牌控場所致

結論: 這場不是模型打錯, 是「真人怪物牌 + self 普通牌」被輾壓。
→ 支持「general 強度天花板」診斷, 而非「可修復的戰術 bug」。
→ 含意: 從線上資料「找弱點」payoff 可能有限 — 輸是因為被輾壓, 不是 exploitable bug。
   真正的槓桿仍是「把網路練更強」(需算力投入), 或接受競爭級但非壓制的結果。

## V7 失敗實驗 (2026-06-07): value-coef=3 適得其反

**動機**: 診斷發現 v_loss(~0.02)被 p_loss(~2.7)淹沒, value 拿到 <1% 梯度。
假設「放大 value loss 權重 → value 練得更準」。配方:全新 600 iter, --value-coef 3.0,
--league --league-ratio 0.3, dominance 內建(306維), forced-pass 跳過。--no-resume。

**結果(同條件對照 V6, probe_value.py + eval_reward.py)**:
| 指標 | V7 (coef=3) | V6 (coef=1, 現役) | 贏家 |
|------|-------------|-------------------|------|
| MCTS-80 vs heuristic (200局) | +4.10 ± 1.60 | **+8.26 ± 1.98** | V6(近2倍)|
| MCTS win% | 22.5% | **30.0%** | V6 |
| greedy avg_score (2000局) | -3.20 | **-2.24** | V6 |
| value 相關 r (晚期) | 0.30 | **0.50** | V6 |
| value 校準斜率 (晚期) | +0.45 | **+0.93** | V6 |

**全面落敗。value-coef=3 不是把 value 練準, 是放大對「高雜訊 MC target」(std~19)的
擬合** → v_loss 訓練中一路 0.015→0.10 上升(target drift + 過度擬合雜訊), 校準反而崩壞。
連帶 policy 梯度被餓瘦 → greedy/MCTS policy 也變差。

**教訓**:
1. 「<1% 梯度」不代表「該放大權重」。Big2 的 MC value target 本質噪音極大, value head
   理應保守(對噪音目標回歸均值, 斜率<1 是正常的)。V6 的 coef=1 已是好平衡。
2. 想練更準的 value, 正確方向不是加權重, 而是「降低 target 噪音」(更多 sims 的更深搜尋
   產生更穩定的 leaf 評估; 或 TD bootstrap), 或更多資料。
3. value-coef 若要動, 上限 ~1.5; coef=3 確定過頭。
4. 再次驗證「乾淨單次 cosine 配方」難被小調參超越 — 跟 re-warm LR 教訓一致。

**結論: V7 棄用。V6 維持現役(全程未動)。下一次迭代應投資已記錄的大槓桿
(更多 self-play sims、約束式 determinization、真人棋譜), 不是繼續微調係數。**
