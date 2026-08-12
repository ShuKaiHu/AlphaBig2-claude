# planner/ — Big2MDP 忠實復刻(ToG 2025)

> **使用者裁定(2026-08-12)**:完整復刻 Chen & Lu 的 Big2MDP 整隻 agent,
> **不得使用任何舊資產**(policy/belief/value/AW-BC/人類資料)——agent 全程自己打,
> 只依賴遊戲引擎規則與自打對局統計。研讀筆記:`docs/paper/BIG2MDP_METHOD_NOTES.md`(PR #5 分支)。
> 同時裁定:AB2/AB3 線上戰役棄用(見 STATUS.md)。

## 結構

| 檔案 | 對應論文 | 內容 |
|---|---|---|
| `big2mdp/features.py` | eq 17 特徵、2022 §III 牌值 | 四特徵(首出牌組/Table-Card Level/對手剩牌/自家剩牌)、動作抽象鍵 |
| `big2mdp/store.py` | eq 3, 17-18 | 統計表(N 計數 → P_win/R_win/P_lose/R_lose/R_c),MP 特徵-OR 檢索,存/載 |
| `big2mdp/agent.py` | eq 4-16, 19-27 | 四個難度(MDP 1.0–4.0)、S_end(4/4/30)、WC(.8/.2/.8)、SP(.8/.1) |
| `big2mdp/selfplay.py` | 「500K 自打」 | 四座位共用一表的自打填表迴圈 + CLI(斷點續填) |
| `decompose.py` / `control.py` | SP 的手牌分解 / R_c 精算 | planner 自有元件(非舊資產);control 提供 exact-floor(可 `--no-exact-floor` 關閉走純表) |
| `test_planner.py` | — | 9 個測試(規則語義 + store roundtrip + 自打煙霧) |

`big2mdp/data/` 為 gitignored(表是可再生工件)。

## 現況(v0.2,2026-08-12,誠實記錄)

- ✅ 管線端到端可跑:39 局/秒(單核)→ **500K 局 ≈ 3.5 小時**,本機可行:
  `python -m planner.big2mdp.selfplay --games 500000 --level 4 --save planner/big2mdp/data/store_l4.pkl`
- ⚠️ **已知忠實度缺口(下一個里程碑)**:目前決策核心用「一步聚合統計」選動作;
  論文的核心是**路線/預測樹**——從當前狀態沿預測的對手動作展開後繼狀態,reward 自終局
  收斂回來(eq 5-6 的 V/Q 遞迴)。煙霧證據:6k 局表的 MDP4.0 反而輸給冷啟動規則
  (vs Smart:−3.1 vs −0.1;一步邊際統計把「出 2/炸彈與贏的相關」當因果,提早燒強牌)。
  加了 min_support 護欄仍不夠 → 缺的是樹,不是資料量。
- 煙霧數字(300 局,僅供管線健檢,非效果宣稱):cold vs Random 0.69/+16.2、vs Smart 0.21/−0.1;
  trained(6k) vs Random 0.40/+6.4、vs Smart 0.06/−3.1。

## 路線圖

1. **[下一步] 路線樹**:以 store 的 MP 檢索建 prediction tree(對手動作 → 後繼狀態 → 遞迴到終局),
   V/Q 按 eq 5-9 收斂;WC/SP 掛在樹值上(V4.0_action = Σ R_c over C(s,a),eq 23)。
2. 樹完成後重跑訓練曲線(10K/50K/100K/500K),對照論文 Fig 13-14 的爬升形狀驗證忠實度。
3. 評估(Random/Greedy/Smart 是 Patwa 尺,僅作 yardstick)→ 事前註冊 → 線上。

## 已記錄的實作假設(論文未明定處)

牌值 = rank(1-13) + 花色加值(♣1♦2♥3♠4),level = 累計//10;動作以(張數,型,關鍵 rank/窗)抽象;
store 用「逐特徵聚合表求和」近似 eq 17 的集合聯集(多特徵吻合的紀錄被多算,權重≈相似度);
冷啟動 = 出最大張數中最低值的合法組;MDP3.0 的 d = 貪婪分解組數;min_support=32 護欄
(表大後無作用)。每一條都要在 paper 的 baseline 描述裡如實揭露。

## 紀律

評估宣稱必須事前註冊、走 V0–V3 驗證梯;gate 尺凍結;本 README 的煙霧數字不是結果。
