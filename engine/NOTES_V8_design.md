# V8 設計文件 (2026-06-08)

> 前提:V7(value-coef=3)已認定失敗(離線全面輸 V6;checkpoint 已遺失)。
> V8 = 兩個並行、可各自衡量的改動。最終成敗以**使用者線上對真人實測**為準
> (見 memory `eval-needs-online-data`),離線指標只是篩選器。

---

## 反省總結:還沒解決的兩個真問題

1. **value target 太吵(離線可修)**:`self_play.py:202,211` 把「整局共用的終局分數」
   `value_target_vec` 給每一步當 target → 開局狀態被要求預測 ±19 變異的最終分,
   不可能 → 早期 value corr 僅 0.13(V6/V7 皆然)。V7 用 coef=3 蠻力壓,適得其反。
   **正解 = 降低 target 變異(bootstrap),不是加權重。** `LAMBDA_TD=0.9` 早已備好卻沒接上。

2. **對真人的不完全資訊推理(線上、self-play 學不到)**:self-play 是完美資訊 MCTS,
   網路從沒在「不知道對手牌」下被訓練。真人靠 pass 揭露的 void + dominance 推理。
   這是「打平→贏」的真正差距,且**只對真人有意義**(對 heuristic 無差,已實證)。

---

## 🅰 TD(λ) value target(V8 的網路改動)

### 現況
```python
# self_play.py
value_target_vec = np.tanh(terminal_rewards / REWARD_SCALE)  # 4-dim, 絕對玩家索引
# ... 整局每一步都用同一個 value_target_vec.copy()
```
每個 state 都預測同一個終局結果 → 早期 target variance 最大化。

### 設計:forward-view TD(λ),terminal-only reward
大老二只有終局有 reward、無中間 reward、γ=1。標準 TD(λ) forward view 在此化簡為
「網路自己對後續狀態的估值」與「真實終局」的 λ 加權混合:

對「依出牌順序收集的」步序 i = 0..M-1(4 維、絕對玩家索引;value head 本來就輸出
絕對玩家 reward,跨步可直接混合),令 `v_i` = 該步網路(或 MCTS root)的 value 估計,
`G` = 終局 tanh 4-vec。後向遞迴:
```
target[M-1] = G                                  # 最後一步 → 直接 bootstrap 到終局
for i in range(M-2, -1, -1):
    target[i] = (1 - λ) * v[i+1] + λ * target[i+1]
```
- λ→1 ⇒ 退回現況(純 MC,高變異);λ=0.9 ⇒ 大幅平滑早中期 target。
- 直覺:早期狀態不再背負「整局運氣」,改為「下一步的合理估值 + 一點終局訊號」。

### 實作要點
1. **存 `v_i`**:每個收集步多存一個 value 估計。最省成本 = 用 MCTS root 的 backed-up
   value(搜索改良過、比 raw network 更準);退而求其次用 `model.forward` 的 value。
   `MCTS.run` 需回傳 root value(目前只回 action+visits)。
2. **跳過步的處理**:`run_episode` 會略過 forced-pass 與 frozen-opponent 步,trajectory
   有缺口。v1 近似 = 只在「收集到的步」之間做遞迴(value 是緩變的狀態屬性,近似可接受)。
   若不夠好,v2 再對所有 env step 存 value。
3. **絕對玩家索引一致性**:`v_i` 與 `G` 都是 4-dim 絕對玩家向量,遞迴逐維獨立、無視點翻轉
   (沿用已修好的 max^n 無翻轉慣例)。
4. **早期 bias 防護**:訓練初期網路 value 是垃圾,bootstrap 會注入偏差。
   作法:**前 N iter(例 ~bc-warmup 之後再 +20)用 λ=1(=純 MC),之後切到 λ=0.9**。
   或直接固定 λ=0.9(因偏高、偏向 MC,初期風險有限)。先試固定 0.9,不行再退火。
5. **不動其餘**:sims=50、value-coef=1、一次性 cosine、dominance 內建、league 視情況。
   **只改 value target 這一個變數** → 可乾淨歸因(V7 的最大教訓)。

### 成功判準
- 離線:早期 value corr 從 0.13 顯著上升、晚期校準斜率趨近 1;**MCTS-80 vs heuristic ≥ V6 的 +8.3**。
- 線上(最終):使用者跑 ≥30 局,avg_score(server 分)≥ V6 線上水準。
- 成本:**與 V6 同速**(~110s/iter,600 iter ≈ 18h)。

### 風險
- bootstrap 在弱網路初期注入偏差(用 λ 退火緩解)。
- value 變了會透過 MCTS leaf eval 影響 policy;這是預期的耦合,用 MCTS-eval 總體衡量即可。

---

## 🅲 線上 void-constrained determinization(不重訓,平行)

### 現況
`alpha_big2_wrapper.py` 的 `_sample_opponent_hands`:在未見牌中**均勻隨機**分給 3 家
(只尊重已知張數)。完全忽略出牌過程揭露的資訊。

### 設計:用「pass 揭露的硬約束」排除不可能世界
理性假設下,某家在「有機會壓」時 pass 掉某型某 level → 他**沒有更大的該型**。
逐手累積成 per-opponent void,採樣時**硬性排除**違反 void 的發牌。

### v1 範圍(先做最高訊號、最可靠的)
- **單張 void**:跟牌階段對「單張 level R」pass → 該家 void「單張 > R」
  → 採樣時不發「會成為其可出單張且 > R」的牌給他。
- **對子 void**:同理對「對子 > R」。
- 五張型(順/葫蘆/炸)的 void 較複雜(牽涉組合),v2 再做。

### 與 belief 的關鍵區別(NOTES 已實證)
- ❌ 軟性 belief 機率加權採樣:用目前弱 belief → 偏向錯誤世界 → **已實證有害**。
- ✅ 硬邏輯排除(void):理性推論的確定性約束,只排除「不可能」世界,不偏移分布 → 安全。

### 實作要點
1. 維護 `opp_voids[seat] = {"single": R, "pair": R, ...}`(各型的「壓不過」上限)。
2. 每手更新:偵測「該家有合法更大出牌機會卻 pass」→ 更新對應 void。
   注意只在「非被迫 pass」時更新(被迫 pass 不揭露資訊)。
3. `_sample_opponent_hands` 改成**約束式採樣**(rejection 或 constrained assignment):
   發牌時略過會違反 void 的牌;若卡死(約束過嚴)則放寬退回均勻。
4. 適用於 V6 與未來 V8(純線上 inference 層,與網路權重無關)。

### 風險 / 注意
- **真人會詐唬**:偶爾握著贏牌仍 pass → 我們會誤排除 → 可能誤導 MCTS。
  但平均而言 pass 多為「真的壓不過」;先上線實測淨效果。若有害就退回均勻(像 belief 那樣果斷停損)。
- 只對「出牌有資訊量」的對手(真人)有價值;對 heuristic 幾乎無差(別用 heuristic 評估它,
  要看線上真人數據)。

### 成功判準
- 線上:使用者實測,出完率↑ / 災難場↓ / avg_score↑。**只能線上驗證**(離線 heuristic 測不出)。

---

## 執行紀律(來自全專案教訓)
- **一次一個變數**:🅰 與 🅲 正交(訓練 vs 推理),可各自衡量、不互相混淆。
- 🅰 用獨立 `--checkpoint-dir engine/checkpoints_v8`,**絕不覆蓋 V6**(best.pt/latest.pt/saved/)。
- 🅒heckpoint 要及早複製進 git-tracked 的 `saved/`(避免重蹈 V7 被誤刪覆轍)。
- 期待務實:這是有原理的精進,非保證突破;天花板若是規模本質,真正破關需更大算力或真人棋譜。

---

## 🅰 結果 (2026-06-09, 600 iter 跑完, checkpoints_v8/best.pt = saved/v8_td_deploy.pt)

**訓練動態:v_loss 全程穩定** 0.011→0.041(對照 V7 純MC+coef=3 失控到 0.10)→ TD 確實降了
target 變異,沒搞砸。training best greedy avg_score = 1.57(V7 是 0.51)。

**離線評估(probe_value.py 400局 + eval_reward.py MCTS-80 200局, 同條件對照 V6):**

| 指標 | V8 (TD) | V6 (MC) | 判讀 |
|------|---------|---------|------|
| MCTS-80 vs heuristic | +5.30 ± 1.80 | **+8.26 ± 1.98** | 點估計偏 V6,但 ~1.1σ **不顯著**(CI 大幅重疊)|
| MCTS win% | 27.0% | 30.0% | — |
| value corr 整體 | **0.350** | 0.318 | 略升 |
| value corr 晚期 | **0.542** | 0.495 | 升 |
| value corr **早期** | 0.135 | 0.133 | **沒動 ← 核心假設失敗** |
| value 校準斜率 晚期 | 0.81 | 0.93 | V6 較佳 |

**結論:**
1. TD value **沒害也沒明顯助益**。離線 V8 與 V6 **統計上分不出高下**(點估計略偏 V6)。
   ≠ V7(V7 是 value 崩壞 + 全面明顯變差);**V8 是乾淨、同檔次的模型**。
2. **核心假設「TD 修早期 value」失敗**:早期 corr 仍 0.13。
   研判**早期 value 瓶頸是「資訊」非「噪音」**——隨機發牌開局本就難預測結局,降變異救不了。
3. 含意:value target 變異不是現階段的主要瓶頸。下一步不該再碰 value target;
   真正的槓桿仍是 NOTES_belief 記錄的:**更多 sims、約束式 determinization、真人棋譜模仿**。
4. **最終成敗仍待使用者線上實測**(memory `eval-needs-online-data`):離線分不出 →
   V8 是「乾淨、可上線一試」的候選,但無離線證據預期它會贏 V6。
   V6 維持現役未動;V8 存於 saved/v8_td_deploy.pt。

## 🅰 線上實測結果 (2026-06-10) — 最終裁決:V8 線上優於 V6

使用者線上對真人實測(reward_log.jsonl,server 分數):
| | 局數 | 平均 | 95%CI | 中位 | 出完 | 最差 |
|---|------|------|-------|------|------|------|
| **V8** | 123 | **+1.80** ± 1.58 | [-1.3,+4.9] | -3.0 | 29% | **-28** |
| V6 | 120 | -2.26 ± 1.44 | [-5.1,+0.6] | -4.0 | 24% | -80 |

差 V8−V6 = **+4.06, ≈1.90σ**(67局時 1.63σ → 樣本增大訊號收緊,非噪音)。

**重大結論:離線與線上「排名相反」。**
- 離線對 heuristic:V6(+8.26)> V8(+5.30)。
- **線上對真人:V8(+1.80)> V6(-2.26)。**
→ 鐵證:**對 heuristic 強 ≠ 對真人強**,離線指標會選錯模型。驗證 memory `eval-needs-online-data`
  的正確性 —— model 成敗只能線上定。
→ V8 災難場大減(最差 -28 vs -80),推測 value 校準較好(晚期 corr 0.54>0.50)→ 劣勢局不崩。
→ TD(λ) value 雖在「離線 heuristic / value-corr」上看似平平,**對真人卻是真進步**。

**注意 caveat**:1.90σ 差 0.1 到嚴格顯著;V8 自身 avg+1.80 僅 1.14σ>打平(還沒「穩定贏錢」);
V8/V6 不同天跑、對手可能不同(confound)。但 V8「最差平 V6、很可能更好、風險更低」,扶正下檔風險小。
