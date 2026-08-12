# Paper 計畫(定位、主張、骨架、結果總帳)

> 建立:2026-08-12。前情:使用者決定以本專案寫論文;定位討論見本檔 §1。
> 相關檔:`OUR_RULES_AND_ARCH.md`(我方規則/架構正典)、`RELATED_WORK_COMPARISON.md`(文獻比較)。

## 1. 定位(已選:Framing B)

**不是**「又一個大老二 AI」(Charlesworth 2018、Patwa 2026、Big2AI 2022、MDP 2024、DeepMC 2024 已做過;
見 `ppo/SURVEY_evaluation.md`)。

**是**:方法論/案例研究——
> 在只有便宜離線代理指標、線上評估又貴又小樣本的環境裡,
> 如何科學地迭代一個要在真實平台上打贏真人的 agent。

自己調查已指出的文獻缺口(SURVEY takeaway #4):**不存在嚴謹的強真人大老二基準;做出來就是貢獻**。

### 三條貢獻主張(草案)

1. **基準/系統**:活平台(神來也)對排位真人的評估管線——6,225 局全資訊真人資料集、
   seat 標記、去污染規則(丟伺服器自動 pass、bomb-free 分層)、線上 A/B 基礎設施(腦/身雙 repo)。
2. **方法論(靈魂)**:小樣本線上評估下的迭代紀律——300 局/臂只測得出 ≥4.5 分 →
   主終點改用**機制指紋**(領五率/囤2率)當高統計力代理;運氣基準;配對牌局;
   **事前註冊判讀規則**;凍結 gate 尺(`eval_policy_gates.py`)取代 val_acc 選型。
3. **實證發現(含 null)**:self-play ≠ 打贏真人;belief 學得起來但轉不成強度(×2 嘗試);
   KL 錨定 RL 對指定機制(囤2)乾淨 null(連訓練集都沒動)。有事前註冊撐腰,null 是發現。

備選 Framing A(強度宣稱型)僅在 AB2/AB3 之後某方法真正關掉缺口時升級;照 B 寫不白工。

## 2. 骨架(按主張組織,不按時間軸)

1. Intro:北極星(線上對真人 avg_score,非 win_rate)+ 缺口分解(−7.7/局,勝率通道 77%)
2. Related work:規則變體表 + 架構比較表(`RELATED_WORK_COMPARISON.md`)+ 評估方法調查(`ppo/SURVEY_evaluation.md`)
3. 系統:規則/平台(`OUR_RULES_AND_ARCH.md` §A)、資料收割、腦/身架構
4. 模型:card-aware 雙塔 + 歷史 GRU(`OUR_RULES_AND_ARCH.md` §B)、BC → AW-BC → 錨定 RL 方法梯(M1–M4)
5. 評估方法論:驗證梯 V0–V3、機制指紋、事前註冊、運氣基準、污染控制、量測時代標記
6. 結果:結果總帳(§3)挑 3–5 條扛得住的主張,**含 null**
7. 教訓/討論:self-play 外星風格、belief-強度轉換失敗、argmax 坍縮、val_acc 陷阱

### 投稿場子候選
IEEE Transactions on Games / IEEE CoG / AAMAS;先 workshop 版亦可。

### 投稿前要決定的兩件事
- **倫理/平台**:商業平台上 bot 對真人(虛擬幣、非真錢);論文中平台匿名化與資料處理聲明。
- **樣本誠實**:最好乾淨時代線上數字 −1.13(n=69,CI 碰損益平衡);對照 Charlesworth 664 局標準,不超賣。

## 3. 結果總帳(ledger)——待逐條填

> 規則:一行一主張;證據、n、是否事前註冊、**量測時代**(parser 修正前後、voluntary-pass 修正前後、
> 結算倍數修正前後、bomb 指標只認 post-fix)。跨時代數字不可直接比,論文須標記。

| 主張 | 證據 | n | 事前註冊? | 時代 | 狀態 |
|------|------|---|-----------|------|------|
| self-play 系(V1–V5)對真人全輸但缺口減半(−9.1→−3.78) | 線上 V1–V5 各 ~50 局 | ~250 | 否 | 舊(parser 修正前) | 待核 |
| V5(V4+search)= 當年最佳(−3.78,win 18%) | 線上 50 局 | 50 | 否 | 舊 | 待核 |
| belief 可學(P@count 82.8%)但轉不成強度 | belief-RL null;belief-guided PIMC 44 局 −10.23 vs −3.78 | 44 | 否 | 舊 | 待核 |
| V9a(全資訊 value + 約束 determinization)−1.13,CI 碰平 | 線上 69 局(clean parser) | 69 | 部分(離線 4 gate 事前) | clean | 待核 |
| 純 BC(policy_4500)缺口 −7.7/局;勝率通道 77% | 線上 ab1 control | 300 | 是 | 現行 | 待核 |
| G1 強牌轉換/G2 輸勢囤2 病灶定位(vs 真人錨點) | bomb-free 分層統計 + 8,555 局面 last-chance 正典 | — | 是(指標凍結) | 現行 | 待核 |
| D2-via-RL = NULL(錨定 PPO + 50% 注入動不了囤2) | 34 保留位置 64.5% vs 64.0%;訓練集 286 位置同樣沒動 | 30 reps | **是** | 現行 | 已裁決 |
| vs-pool 選型 > vs-Smart / min-loss | pool_registry 紀錄 | — | 部分 | 現行 | 待核 |
| AB2 指紋轉移(AW-BC T40) | 三臂各 300 局 | 900 | 是(腳本頭事前規則) | 現行 | **跑完待判讀** |
| AB3(m3p1_200 vs 400) | 兩臂各 300 局 | 600 | 是 | 現行 | 排程中 |

> 待辦:回到本機 session 後,把 AB2/AB3 判讀結果與 `ppo/data/experiment_logs` 各 result commit
> 逐條補進來,再挑論文主張。
