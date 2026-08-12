# 本專案的規則變體與模型架構(paper「我方欄位」正典)

> 目的:related work 比較表的「本專案」欄。每一條都對應到 code 出處,寫論文時直接引用。
> 建立:2026-08-12(paper 籌備 session)。規則 = 神來也大老二(gamesofa)台灣規則,
> 引擎結算已逆向工程到對平台 showScore **4412/4412 全吻合**(`big2Game.py::_winner_finish_multiplier` docstring)。

## A. 規則變體(我們實作/部署的版本)

| # | 規則點 | 本專案(神來也/台灣) | 出處 |
|---|--------|----------------------|------|
| A1 | 人數/發牌 | 4 人 × 13 張 | `rules.md` |
| A2 | 點數序 | 3<4<…<K<A<2(2 最大) | `rules.md` |
| A3 | 花色序 | **♣ < ♦ < ♥ < ♠**(梅花最小) | `rules.md` |
| A4 | 起手 | 持**梅花 3** 者先出,首手必含梅花 3 | `rules.md`、`big2Game.py`(mustPlayClub3) |
| A5 | 合法牌型 | 單張、對子、五張(順子/葫蘆/鐵支+1/同花順)。**無三條、無兩對、無同花(flush)** | `rules.md`、`enumerateOptions.py` |
| A6 | 順子窗 | **恰 10 窗**:3-4-5-6-7 … 10-J-Q-K-A(8 窗)+ A-2-3-4-5 + 2-3-4-5-6;**J-Q-K-A-2 / Q-K-A-2-3 / K-A-2-3-4 不合法** | `value/hand_features.py::STRAIGHT_WINDOWS_RIDX`(import-time assert) |
| A7 | 順子排序 | A-2-3-4-5 最小 … 10-J-Q-K-A 次大,**2-3-4-5-6 最大**;同窗比最高牌花色 | `rules.md` |
| A8 | 越級(炸彈) | 鐵支可壓任何當前牌型(除同花順/黑桃2單張);同花順可壓任何(除黑桃2單張);**黑桃 2 單張無敵**(全場必 pass) | `rules.md`、`big2Game.py::returnAvailableActions` |
| A9 | Pass 規則 | **能壓也可自願 pass**;pass 後本 trick 鎖定;**連續 3 pass 清 trick**(含伺服器自動 pass) | `big2Game.py`(passedThisRound)、CLAUDE.md 鐵律 |
| A10 | One-card rule | 下一個「仍活躍」行動者只剩 1 張時,我的單張選項被限制為**只能出最大的合法單張**(領出與跟牌皆適用;只看第一個活躍座位,不掃過緩衝座位) | `big2Game.py::_restrict_singles_for_one_card_rule` |
| A11 | 基礎計分 | 輸家 = −自己剩牌數;贏家 = 收三家總和 | `big2Game.py::assignRewards` |
| A12 | 輸家倍數 | 手上每張 2 → ×2(n 張 = 2^n);手上有鐵支 → ×2;手上有同花順 → ×2;**剩牌 ≥10 張 → ×2**(各項相乘) | `big2Game.py::_hand_multiplier` |
| A13 | 贏家收尾倍數 | 最後一手是炸彈(鐵支/同花順)→ 全桌 ×2;最後一手含「**主要作用的 2**」→ 再 ×2(可疊成 ×4)。**附帶的 2 不算**:A-2-3-4-5 的 2、葫蘆的對 2、鐵支的 2 kicker | `big2Game.py::_winner_finish_multiplier`(4412/4412 驗證) |

**論文可主張**:這是文獻中未被使用過的規則變體組合(無 flush、繞順 + 2-3-4-5-6 最大、
黑桃2 無敵、炸彈越級、one-card rule、乘法結算),且以真實商業平台的結算為 ground truth
逐局驗證——不是自訂簡化規則。各文獻用的變體見 `RELATED_WORK_COMPARISON.md`。

## B. 模型架構(部署主線:`CardAwareActorCriticHistory`)

檔案:`ppo/network_cardaware_history.py`(繼承 `ppo/network_cardaware.py` 的設計)。
訓練:`train_policy_history.py`(BC / AW-BC)、`ppo/ppo_trainer.py --arch cardaware_history`(RL 線)。

### 三個使用者關心的軸

1. **History 作為 input** ✅
   - 每局完整出牌史,上限 **196 步**(49 手 × 4 座);每步 **108 維原始編碼**:
     座位 one-hot(4)+「當時必須回應的檯面牌」raw 52 multi-hot +「實際打出的牌」raw 52 multi-hot
     (pass = 全零 played;**pass 事件保留**——「拒絕壓 X」是 belief 的核心訊號)。
   - 序列經 **GRU(hidden 128)**,末 hidden state 併入 state 塔。
   - 出處:`ppo/belief_history_dataset.py`(HIST_STEP_DIM_V2 = 4+52+52,HISTORY_LEN_V2 = 196)、
     `network_cardaware_history.py::steps_from_action_history`(trick 追蹤 = 連續 3 pass 重置,排除 forced_skip)。
   - ⚠️ **誠實描述**:歷史編碼器是 **GRU,不是 transformer**。

2. **Transformer / self-attention** ⚠️(部分)
   - **手牌**編碼:53×32 共享 card embedding → **單層 4-head self-attention**(`nn.MultiheadAttention`)
     → masked mean-pool 成 hand vector。
   - 這是「一層 self-attention」,**不是完整 transformer encoder**(無 FFN block、無殘差堆疊、無 positional encoding——手牌是集合,本來就不需要位置)。論文寫法建議:
     "a single multi-head self-attention layer over card embeddings (set encoder)",不要寫 "Transformer"。
   - 同一 pattern 也用於 belief(`belief_model_history.py`)與 value 線(`value/value_model.py` 等)。

3. **雙塔(two-tower)動作打分** ✅
   - **State 塔**:concat[hand_attn_vec, trick_vec, seen_vec, 3×opp_played_vec, opp_counts, pass_count, GRU history context] → MLP → S ∈ R^128。
   - **Action 塔**:每個「**合法**」動作的預計算 64 維特徵(出牌 52 multi-hot + 型別 one-hot + 五張子型 + top rank + 張數 + 2 張數 + is_pass,`ppo/action_features.py`)→ MLP → A_i ∈ R^128。
   - **打分**:score_i = (proj(S) · A_i) / √128,softmax 只在合法動作上;critic 是 S 上的 MLP value head。
   - 核心動機:14739 動作中 ~70% 是順子的花色變體;固定輸出頭把它們當 14739 個無關 slot,
     雙塔讓近似動作共享學習,且**只給合法動作打分**(不是 mask 掉的固定頭)。

### 沿革(論文的 architecture ablation 素材)

| 世代 | 網路 | 歷史 | 備註 |
|------|------|------|------|
| legacy AlphaZero 線 | `engine/model.py` Big2Net | GRU(128) | ResBlock trunk ×4;policy(14739 固定頭)+ value(4 維)+ belief 三頭;MCTS 用 |
| PPO 線 baseline | `ppo/network.py` MLP | 無 | 412-bit 狀態 + 14739 masked 頭(~8.0M 參數) |
| card-aware | `ppo/network_cardaware.py` | 無 | 雙塔 + 手牌 attention(~0.1M 參數);`train_bc.py` 用此 |
| **card-aware + history(部署)** | `ppo/network_cardaware_history.py` | GRU(196×108) | 上表 + belief 同款歷史編碼;policy_4500 / AW-BC / m3p1 線皆此 |

### 相對文獻的定位(等 research 結果後在 RELATED_WORK_COMPARISON.md 定稿)

- 雙塔/合法動作打分的設計跟隨 Patwa 2026(本專案 `ppo/README.md` 明載);**歷史 GRU 塔 + 原始
  required/played 逐步編碼(含 pass 事件)是本專案加上去的**,動機是 belief:pass 揭露的約束
  只存在於跨 trick 歷史中。
- 「動作當輸入」在鄰近文獻有前例(DouZero 的 Q(s,a) concat 式),但 dot-product 雙塔 + 只列舉
  合法動作 + 歷史塔的組合是否有前例,見比較表。
