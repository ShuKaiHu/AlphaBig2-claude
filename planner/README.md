# planner/ — Big2MDP 忠實復刻(ToG 2025)

> **使用者裁定(2026-08-12)**:完整復刻 Chen & Lu 的 Big2MDP 整隻 agent,**照抄論文、不自行改設計**,
> 不得使用任何舊資產(policy/belief/value/AW-BC/人類資料)。研讀筆記:`docs/paper/BIG2MDP_METHOD_NOTES.md`。

## 結構(v4,paper-literal)

| 檔案 | 對應論文 | 內容 |
|---|---|---|
| `big2mdp/features.py` | eq 17 特徵、2022 §III 牌值 | 四特徵、動作抽象鍵(PASS 是普通動作) |
| `big2mdp/store.py` | eq 1-3, 17 | **RecordStore**:逐決策原始紀錄 + 同座位後繼鏈(succ)+ npass(0-3,eq 19-20 字面)+ 特徵倒排索引;檢索 = 特徵-OR 聯集 |
| `big2mdp/tree.py` | eq 3-9, 14-16, 19-20 | **預測樹字面版**:檢索群 → 依動作分群 → 後繼依狀態簽名分支 → 機率 = 出現頻率 → 自家節點取 max → reward 自終局收斂;產出 p_win / q1(max-of-products)/ loss / rc / d_min |
| `big2mdp/agent.py` | eq 4-16, 19-27 | 四難度 = MDP 1.0-4.0;S_end(4/4/30)、WC(.8/.2/.8)、SP(.8/.1);優先序 SP→WC→頭 |
| `big2mdp/selfplay.py` | 「500K 自打」 | 共庫自打;每局結束整批入庫(succ 鏈 + npass),玩一場學一場 |
| `decompose.py` / `control.py` | SP 手牌分解 / 測試輔助 | control 僅供測試與規則驗證,agent 不再使用(v3 的發明已全數移除) |

`big2mdp/data/` gitignored。v3 的自創設計(存活折扣、rc 先驗、PASS 特殊定價、exact-floor)已依使用者
「照抄論文」指示**全部移除**;PASS 經同一棵樹評估(論文動作集 = {Play, Pass})。

## 論文沒寫、抄不到的地方(最小假設,全部列出)

1. **檢索上限 cap 與樹深 depth_max**:論文無界;但成本隨資料庫成長(2454ms/手是滿庫量測),
   任何實作都需要邊界。訓練組態 cap=200/depth=5(8.1 局/秒,50 萬場 ≈ 17.5h);
   部署/評估可放大(論文出手限時 10 秒)。
2. **冷啟動**(檢索為空):出最大張數中最低值的合法組(論文未載 bootstrap)。
3. **V_cover/V_series 的 Σ 讀成平均**:eq 21/23/25 字面是對狀態集合求和,與 0.8/0.1 閾值量綱不合,
   唯一可讀解釋是機率(均值)。
4. **損失側傳遞**:遞迴中 loss 沿「勝率最大化動作」傳遞(論文未指明)。
5. **動作以(張數,型,關鍵 rank/窗)抽象**:具體牌組合表會稀疏到查不到;論文的 MP 特徵配對本就在
   這個粒度運作。
6. **訓練期組態**:論文未說明訓練局用什麼打(滿樹速度物理上跑不完 50 萬場;
   庫小時快、庫大時慢——cap 等效於把成本凍結在早期水準)。

## 驗證計畫

訓練曲線 = 忠實度主檢驗:10K/50K/100K/500K 存檔點評估勝率是否如論文 Fig 13-14 爬升。
評估宣稱須事前註冊、走 V0-V3 梯;Random/Greedy/Smart 只是 Patwa 尺(yardstick),不是方法一部分。
