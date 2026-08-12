# planner/ — Big2MDPLite(規劃器對照臂)

> 靈感:Chen & Lu, *Big2MDP*(IEEE ToG 2025)的 SP(連出收尾)與 Send(局末止損)機制;
> 研讀筆記與借鏡分析見 `docs/paper/BIG2MDP_METHOD_NOTES.md`(PR #5 分支)。
> 定位:policy/heuristic 臂 + 兩個**規則 override**,無需訓練;作為線上 A/B 的對照臂與
> G1/G2 機制歸因工具(M3 Phase 1 D2-via-RL 判 null 之後,「規則直給」是最便宜的對照)。

## 元件

| 檔案 | 內容 |
|---|---|
| `decompose.py` | 貪婪手牌分解(與 `value/hand_features.min_plays_to_empty` 同優先序;partition 數 == min_plays) |
| `control.py` | **確定性拿權分析** `guaranteed_hold`:最壞情況下(對手持有全部相關未見牌)此組合是否不可被壓。= Big2MDP 的 R_c,但用規則精算,不是頻率表 |
| `agent.py` | `MDPLite(fallback=...)`:SP 連出(≤1 組非保證 → 依序出保證組收尾)+ Send 止損(局末無勝線且對手將出完 → 先丟 2)→ 其餘走 fallback(預設 Smart;部署目標 policy_4500) |
| `test_planner.py` | 8 個單元測試 + agent 煙霧測試(`python -m planner.test_planner`) |

## 與 engine/dominance.py 的關係(重要)

`control.py` 只 import 它的低階原語(unseen/bomb/SF possible),**判定邏輯自帶修正版**,因為
dominance.py 有兩個已知問題但**不可修改**(其輸出是 v6 已部署 checkpoint 的凍結特徵):
1. `play_strengths:182` 把單張 ♠2 寫死強度 1.0(舊「♠2 無敵」錯誤規則的殘留;引擎本體正確——
   炸彈可壓單張 ♠2,rules.md 修正在 paper 分支 PR #5)。
2. `higher_straight_possible` 用窗頂 rank 值排序,把 2-3-4-5-6(**最大順**)誤排成幾乎最小;
   正確排序 = `gameLogic._STRAIGHT_SEQUENCES` 的 index(A2345 最小 … 23456 最大),
   `control.py` 依此實作並在 import 時與 `value/hand_features` 正典交叉驗證窗集合。

## 現況(v0,2026-08-12,remote session 建置)

- 單元測試 8/8 過;煙霧評估(400 局,Smart fallback):MDPLite vs 3×Smart ≈ Smart vs 3×Smart
  (差異在雜訊內)。**這不是效果宣稱**——overrides 觸發率低,對 Smart 對手本來就不易顯效。
- 預期的真正檢驗:fallback 換 policy_4500 → `eval_policy_gates.py` 量 G1(領五/完整打出)與
  G2(囤2)指紋 → 若指紋動了,才有資格進線上 A/B。

## 紀律(不可跳過)

1. 本臂的任何評估宣稱**必須事前註冊判讀規則**(寫進腳本頭,凍結),走標準驗證梯 V0–V3。
2. gate 尺用 `eval_policy_gates.py`(凍結),不得為本臂另立指標。
3. v1 方向(依序):policy_4500 fallback 接線 → SP 的機率化 R_c(接 belief)→ Send 閾值掃描
  (目前沿用 Big2MDP 調參值 α=4)→ 連出時的出牌順序最佳化。
