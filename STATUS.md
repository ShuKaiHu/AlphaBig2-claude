# 多線狀態板(STATUS.md)

> **新 session 開場程序**:① 讀這個檔 ② 跑 `./status.sh` 對帳(板子可能過期,process 不說謊)
> ③ 接手某條線之前,把「負責 session」欄改成自己的 session id 並 commit。
> **更新紀律**:啟動/暫停/完成任何一條線的 session,必須當場更新本檔並 commit。

_最後更新:2026-07-18 深夜 by session `ef4fa0ae`(本檔建立)_

## 🟢 進行中

### 線 1:AB2 三臂線上測試(P5)
- **狀態**:RUNNING(自動續跑迴圈,瀏覽器崩潰會自癒)
- **負責 session**:`ef4fa0ae`(2026-07-16 起的馬拉松 session)
- **內容**:ctrl(policy_4500)/ T40_last(指紋派)/ T40_best(分數派)各 300 局,50 局腿交錯
- **進度**:C 163+ / L 150 / B 150 → 300;ETA ~2026-07-19 中午前
- **檢查**:`pgrep -f resume_ab2`;各臂局數見 status.sh
- **完賽後**:三臂面板(主指標=線上機制指紋:領五率/囤2率,用正確 3-pass trick 追蹤)→ 指紋轉移裁決 → M3 選型規則定案。事前判讀規則寫在 `Big2VisionAgent-claude/run_awbc_ab2_online.sh` 頭部,不許事後改。

### 線 2b:m3p1_400 續訓(RL Phase 2 前哨)
- **狀態**:TRAINING(從 m3p1_200 熱啟動再 200 updates,seed 8,KL 錨仍 policy_4500;估 ~2h)
- **負責 session**:`ef4fa0ae`
- **完訓後**:①改名 ppo_m3p1_400.pt ②同 8,555 局面跑正典 last-chance 統計出表
  ③AB2 完賽 + 本訓練完成後 → 啟動 AB3(run_ab3_m3p1_online.sh:m3p1_200 vs m3p1_400 各300局,
  事前判讀寫在腳本頭)
- **背景**:正典指標(真人85.7%)已定案;m3p1_200=72.9%(latest 改名;best=upd30 是 500 局小評估的
  幸運峰,已棄用);選型器的雜訊問題記錄在案,Phase 2 要修

### 線 2:M3 Phase 1(錨定 RL 機制探針)
- **狀態**:✅ DONE — **裁決:D2-via-RL = NULL**(照事前規則)。200 updates 完訓,KL 錨全程 0.02-0.05 nats。
- **判決依據**:34 保留位置 sampled 囤2率 64.5% vs 基線 64.0%(argmax 完全相同);**訓練集 286 位置也沒動**(45.1% vs 46.4%)→ 不是背題,是沒學。附帶:policy 整體有學(G4a +0.13→+0.68、領五 +4pp),自然對局 G2 的 52.9→33.3% 是小 n(≈15)+ 選擇效應混淆,依規不採信。
- **後續**:Phase 2 的 D2 目標取消;M3-for-D1(強牌)是否立案等 AB2 指紋轉移裁決。
- **負責 session**:`ef4fa0ae`
- **內容**:單一問題 —— 錨定 PPO + 50% D2 注入,動不動得了囤2率?config:init/KL-ref/對手全 policy_4500、kl-beta 0.02、ent 0.01、luck-baseline on、seed 7
- **主裁判**:`eval_injected_d2.py --ckpt ppo/checkpoints/ppo_m3p1_best.pt --reps 30 --seed-base 0` 對 34 保留位置,vs `ppo/data/injected_d2_baseline.json`(基線 sampled 64.0% / argmax 64.7%)
- **事前判讀(寫死)**:sampled 囤2率 ≤56%(降≥8pp)且 seed 777 復現方向 + gate KPI 無崩壞 → 機制成立進 Phase 2;±5pp 內 → D2-via-RL 記 null;5-8pp → 查 KL/entropy 曲線與注入 transitions 再裁
- **檢查**:`pgrep -f ppo_trainer`;訓練 log `ppo/data/m3p1_train.log` 或 task 輸出;KPI 卡 `tail ppo/data/rl_kpi_log.jsonl`

## ⏸ 待命 / 等待

| 線 | 狀態 | 等什麼 |
|---|---|---|
| M3 Phase 2(完整 RL 配置) | 未開 | Phase 1 機制成立 + AB2 校準選型規則 |
| M2(狀態條件加權 BC) | 未開 | AB2 指紋轉移裁決(轉移才值得做) |
| 收割 flywheel(belief 訓練資料) | 閒置(master 6,225 局) | AB2 完賽後收割新局;belief 是唯一證實的資料槓桿 |
| Chip:RL1/RL2 checkpoint 解剖(gate 工具加 history 架構) | 在使用者手上(task_ff36a986) | 使用者點擊;不擋任何線 |
| Chip:bc_dataset trick 順序疑似 bug | 在使用者手上(task_5f6fae2a) | 使用者點擊 |
| **Big2MDPLite 規劃器對照臂(`planner/`)** | v0 骨架+8 測試完成(branch `claude/big2mdp-lite`,paper session 建置,現無負責 session) | 本機接 policy_4500 fallback → 事前註冊 gate 判讀 → AB2/AB3 判讀後排程;背景見 `docs/paper/BIG2MDP_METHOD_NOTES.md` |

## 🅿 停放(有紀錄、暫不動)

- dominance V2(belief K-world 平均)——「dominance先放一邊」
- value 去飽和(tanh/13)—— 未來 search 線的前提,M1/M3 期間不動
- 炸彈壓制意願 G3 —— 掛監控,等資料累積

## 📌 讀我之前先知道的事(不變的背景)

- 版圖:唯一腦 repo = 這裡(main;value/ 在內);身 = `Big2VisionAgent-claude`。舊路徑 -ppo/-Value 不存在。
- 北極星:線上對真人 avg_score。總缺口 -7.7/局,勝率通道占 77%。
- 兩病灶:強牌轉換(G1)+ 輸勢囤2(G2);根因 = 純 BC 沒看過分數 + val_acc 早停 + argmax 坍縮。
- 詳細戰略記憶在 Claude memory(MEMORY.md index),此檔只管「現在誰在做什麼」。
