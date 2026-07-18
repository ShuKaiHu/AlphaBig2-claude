# AlphaBig2 專案 — session 開場規矩

這個專案**多線並行**(線上測試、RL 訓練、資料收割可能同時在跑,且可能由不同 session 啟動)。

## 每個新 session 開場必做

1. **讀 `STATUS.md`** — 每條線的狀態、負責 session、下一步、事前註冊的判讀規則。
2. **跑 `./status.sh`** — 用 process/檔案實況對帳(板子可能過期,process 不說謊)。
3. 兩者矛盾時**以 status.sh 為準**,並修正 STATUS.md。

## 接手與更新紀律

- 接手一條線之前:把 STATUS.md 該線的「負責 session」改成自己的 session id,commit。
- 啟動/暫停/完成任何一條線:**當場更新 STATUS.md 並 commit**(訊息格式 `status: <線> <動作>`)。
- 別的 session 標記為負責中的線,**不要動**(先問使用者)。

## 不變的鐵律(詳細在 Claude memory)

- 判勝負看線上 avg_score,不是 win_rate;300 局/臂只能測 ≥4.5 分。
- 事前註冊的判讀規則不許看到結果後修改。
- 排除資料只能按「機制」,不能按「記帳」或「結果」。
- 炸彈/順子偵測一律 import `value/hand_features.py` 正典(10 窗;J-Q-K-A-2/Q-K-A-2-3/K-A-2-3-4 非法),不許手寫窗格。
- trick 追蹤必須數連續 3 pass(含伺服器自動 pass)。
- checkpoint 選型用 gate 尺(`eval_policy_gates.py`,凍結不許改指標定義),不用 val_acc。
- 版圖:腦=本 repo(main;value/ 在內;legacy-alphazero 留史);身=`../Big2VisionAgent-claude`。舊路徑 `AlphaBig2-ppo`/`AlphaBig2-Value` 已不存在。
