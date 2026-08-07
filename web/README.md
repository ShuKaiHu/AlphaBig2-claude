# web/ — 瀏覽器人機對戰台

跟 `ppo/play_human.py` 同一個目的(收「人 vs model」的真實 avg_score),但是:

- 走 **engine 線(Big2Net + MCTS)**,因為 deploy checkpoints 就在 repo 裡
  (`engine/checkpoints/saved/*.pt`);ppo 線的 checkpoints 不在版控。
- 瀏覽器 UI,工程師向:所有推論參數可調、policy/value/belief 全部掀給你看。

## 起服務

```bash
# repo 根目錄(需要 torch / numpy / flask)
./.venv/bin/pip install flask        # 只差 flask 的話
./.venv/bin/python -m web.server     # → http://127.0.0.1:8288
# 或
./.venv/bin/python web/server.py --host 0.0.0.0 --port 8288   # 給區網別人玩
```

## AI 決策模式(UI 可選,對應 repo 既有正典路徑)

| 模式 | 對應 | 說明 |
|---|---|---|
| `mcts_det` | `eval_determinization.py` | 決定化 MCTS:**不看**隱藏手牌,取樣 N 個世界跑 MCTS 加總 visits(線上同款)。預設。 |
| `greedy` | `engine/evaluator._model_action` | 純 policy argmax,無搜尋,最快。 |
| `mcts_fullinfo` | `eval_reward.py --mcts` | 全知 MCTS,**AI 偷看所有手牌**,只當 debug/陪練上限用。 |

可調:sims、dets、belief 加權取樣、c_puct、dirichlet(預設 0 = 無探索雜訊)、
V9 全資訊 value net 開關(v9a/v9c checkpoint 才有)、座位、seed、掀牌 debug、
AI 每步 policy top-3 記錄。

## Debug / 分析面板

- **分析(a 鍵)**:model 對你目前局面的 policy top-k(點了可直接照打)、
  4 家 value head、belief head 3×52 熱圖(model 猜誰有哪張牌)。
- **掀牌 debug**:顯示全部四家手牌(state API 才會帶 `all_hands`)。
- **悔棋**:從開局牌組重放到你上一手之前(deal 固定,AI 會重算)。

## 紀錄

完賽自動 append 到 `web/human_games_web.jsonl`(座位、四家分數、cfg、
完整 action 序列、開局四手牌 → 可完整重建對局)。「統計」分頁顯示累計
勝率/平均分。**判勝負看 avg_score,不是 win_rate**(鐵律);單局方差大,
≥100 局才可信。

## 檔案

- `server.py` — Flask 後端;遊戲狀態在記憶體(session id),推論走
  engine 正典模組;不同 checkpoint 的 302/306/312 特徵維度自動偵測
  (鏡射 `eval_reward.load_model`),以 lock 串行化 feature globals 切換。
- `static/` — 無框架前端(index.html / app.js / style.css)。
- `human_games_web.jsonl` — 完賽 log(gitignored)。
