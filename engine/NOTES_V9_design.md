# V9 設計草案 (2026-06-10) — DRAFT,待討論

> 狀態:草案。整合 2026-06-10 的討論(全資訊 value、硬排除 determinization、
> 排除特徵、像人對手、保留並強化 belief)。最終成敗仍以**線上對真人實測**為準
> (memory `eval-needs-online-data`)。動工前先定案要做哪幾個 stage。

---

## 0. 背景:V9 要解什麼

- V8(TD value)線上 +1.80 > V6 -2.26,扶正。但修好 control bug 後,逐手 log
  (`mcts_moves.jsonl`)暴露**核心弱點:跟牌時模型亂送大牌**(62 手跟牌 40% 出得過高,
  用 A 壓 4、2 壓 5)。這是「模型不懂留大牌(控場牌)的價值」。
- 根因鏈(統一洞察):
  1. **value 在 MCTS 裡是「蒙眼」的** —— determinization 已猜定所有手牌,但
     `encode_static` 只餵「自己的手牌」,對手手牌被丟掉 → value 無法判斷「我這張 A 是不是
     非留不可」。
  2. **belief / policy 讀不出對手資訊** —— 因為 self-play 對手是笨 heuristic,pass/出牌
     沒有資訊量,沒東西可推論。
- 結論:V9 從「資訊面」下手 —— 讓 value 在它唯一被使用的場合(determinized、全知)用全知
  方式訓練與評估;並讓「對手知識」以可靠形式進入決策。

---

## 1. 四個組件

### 🅐 全資訊(上帝視角)value net  ★ 核心
- **想法(使用者)**:value 只在 MCTS(已 determinize、假設全知)被用,所以就用全資訊訓練。
  輸入 = 四家手牌(自己 52 + 對手 3×52)+ 公開資訊(誰 pass、要壓單/對、控制權…已在 features)。
- **為何有效**:(a) 直接救跟牌弱點(value 看得到對手最大牌,才知道該不該留 A/2);
  (b) 很可能打破 V8 攻不下的「早期 value corr 0.13 天花板」——那個天花板其實是
  **不完全資訊**造成的(只看自己的牌當然測不準),全資訊下開局可預測性大增。
- **⚠️ 關鍵理論細節(務必注意)**:全資訊 value 在**訓練**時看真手牌,但**推論**時看的是
  determinization **猜**的手牌。若只用「單一隨機猜測」,全資訊 value 會**過度相信一個可能猜錯
  的世界** → 可能比現在「邊際化(只看自己手牌)」的 value 還糟。
  → **全資訊 value 必須搭配「多重採樣平均」**:每步採樣 N 副可能的牌、各自用全資訊 value 評估、
    平均(≈ 對 belief 加權世界求期望)。這才是把 perfect-info value 用在不完全資訊遊戲的正解。
  → 所以 🅐 與 🅑(更好的採樣)是**綁在一起的**,不能只做 🅐。
- NOTES_belief 舊結論「多重 determinization 對 heuristic 沒幫助」是用**舊的邊際 value** 測的;
  配上**全資訊 value**,多重採樣才變成必要且互補。

### 🅑 硬排除(void)determinization —— 不是 soft belief
- **釐清**:使用者要的「下家沒順、上家沒2」是**確定性硬排除**(理性 pass 揭露的 void),
  **不是**神經 belief 的軟機率。
  - soft 神經 belief 引導採樣:**實測有害**(弱 belief 偏向錯誤世界,+4.14 vs 隨機 +12.08)。
  - 硬排除:**永遠 ≥ 隨機**(只砍掉邏輯上不可能的世界,絕不砍錯)。
- **做法**:逐手累積 per-opponent void(對某型某 level pass → 沒有更大的該型;只在「非被迫
  pass」時更新),採樣時排除違反 void 的發牌。已有 v1(`BIG2_VOID`,單張/對子),要:
  - 擴到順子/葫蘆/炸彈(五張型 void,牽涉組合,較複雜)。
  - 與 🅐 的多重採樣結合:每副採樣的牌都合乎 void → 全資訊 value 評估的世界更接近真實。
- **線上實測驗證**(只有真人測得出;heuristic 無差)。

### 🅒 排除/支配特徵直接餵進網路(像 dominance 那樣)
- 不要逼 GRU 自己從原始序列推「下家沒順」;**用邏輯算好 void/exclusion 特徵當輸入**,
  policy 與 value 直接用。可靠、好學。
- 這些特徵**源自公開資訊(passes)**,所以餵給 policy **不算作弊**(policy 仍是不完全資訊)。
- 與既有 dominance(4 維「我這張會不會被壓」)同性質,擴充成一組「對手型別 void」特徵。

### 🅓 像人的對手 self-play(關鍵 enabler)+ 保留並強化 belief
- **統一洞察**:belief 讀對手資訊、policy 讀 history —— 兩者都需要「**值得讀**的對手」。
  對笨 heuristic,沒東西可讀,所以兩者都練不起來。
- **做法**:把 self-play 對手從笨 heuristic 換成「像人」的對手。來源選項(按可行性):
  1. 先用「較強、較像人」的 heuristic(會留大牌、會用 void 推理)當對手 —— 不需資料,先做。
  2. 用真人棋譜訓練的對手 policy —— 需要「真人出牌樣本」pipeline(只需公開狀態→動作,
     不卡隱藏手;見早前討論)。資料目前少(~250 局),要先囤。
- **belief head:保留**。它現在弱是因為對手沒資訊量;有了 🅓 的對手,belief 才有東西可學 →
  變準後可在 🅑 硬排除**之上**做機率細修(Tier B)。順序:先硬排除(可靠),後神經 belief(精細)。
- **policy 已經吃得到 history(GRU)**,架構不用改;配上 🅒 特徵 + 🅓 對手,才會真的學會讀。

---

## 2. 架構:policy 不完全資訊、value 全資訊

- **policy head**:輸入 = 自己手牌 + 公開資訊 + void/dominance 特徵 + history(GRU)。
  **維持不完全資訊**(線上真實決策、MCTS 先驗都靠它,不能讓它看到對手手牌而「作弊」)。
- **value head**:輸入 = 上述 + **對手三家手牌(3×52)**。全資訊。
- **連線兩個選項**(待定):
  - (i) **兩個獨立網路**:最乾淨、好歸因,但參數/算力 ×2。
  - (ii) **共用 history GRU + 公開/自己手牌編碼 → trunk;value head 額外吃一條「對手手牌」
    embedding**:省算力、policy 的 trunk 仍不完全資訊;但耦合較深、較難調。
  - 草案傾向先試 (i) 求乾淨歸因,確認有效再考慮 (ii) 省算力。
- value **target** 維持 TD(λ)(沒壞);全資訊下早期可預測性上升,TD 的角色更輔助。

---

## 3. 分階段(守「一次一個可歸因變數」紀律)

- **V9a(核心,不需真人資料)**:全資訊 value(🅐)+ 多重 void-constrained determinization(🅑)。
  這兩個綁定,是「讓 value 不再蒙眼 + 正確地用全資訊 value」。**直接攻跟牌弱點。**
  - 同時把 🅒 void/exclusion 特徵餵進去(便宜、同向)。
- **V9b**:🅓 換「較強/較像人的 heuristic」對手(不需資料),看是否讓 belief/policy 開始讀對手。
- **V9c(較遠)**:真人棋譜 pipeline → 真人對手模型 → 神經 belief 精細化(Tier B)。需先囤資料。

每個 stage 都用 `mcts_moves.jsonl` 的指標 + 線上實測各自驗證,不混在一起(V7 教訓)。

---

## 4. 成功判準

- **離線(快速篩選)**:
  - **跟牌浪費率↓**(`mcts_moves.jsonl`:有更小合法壓制卻出更高的比例,目前 ~40%)——這是 V9 最直接的指標。
  - 早期 value corr 從 0.13 顯著上升(全資訊應大幅改善)。
  - MCTS-eval vs heuristic 不退步(但記得:對 heuristic 強 ≠ 對真人強)。
- **線上(最終裁決,使用者跑)**:avg_score vs V8 的 +1.80 baseline(用修正版 wrapper 重測),
  災難場、出完率。需足夠樣本(≥100 局)壓低變異。

---

## 5. 風險 / 注意

1. **PIMC 老毛病(strategy fusion)不會被解決**:全資訊搜索本質會假設「能在不同世界選不同手」。
   多重採樣平均緩解,但不根治。可接受(我們本來就是 PIMC)。
2. **全資訊 value 的訓練/推論落差**:訓練看真手牌、推論看猜的手牌 → 採樣品質(🅑)決定落差大小。
3. **架構複雜度↑**:policy/value 分流、多重採樣的算力(每步 N 副牌)→ 線上 1 秒預算要重新分配
   (N 副 × 每副較少 sims,vs 單副較多 sims,要 tune)。
4. **🅓 對手模型資料受限**:真人資料少,V9c 要先囤;V9b 的「較強 heuristic」是不需資料的折衷。
5. **要重新 baseline**:當前部署 = 正確 control + V8;V9 是新架構,別覆蓋 V8(獨立 checkpoint,
   best 及早進 git-tracked saved/,勿重蹈 V7 覆轍)。
6. **野心 vs 務實**:V9a 已是不小的改動;先確認 V9a 在「跟牌浪費率 + 線上」有效,再往 V9b/c。

---

## 6. 待議 / 開放問題

- 多重 determinization 的 N 與每副 sims 怎麼分(線上 ~1 秒預算)?
- 全資訊 value 要不要也在「根節點」用?根是真實決策(對手手牌未知)→ 一樣靠 determinization 採樣平均。
- 兩網路(i)還是共用 trunk(ii)?先 (i)。
- value target:全資訊下要不要從 TD(λ) 退回純 MC(全資訊使終局更可預測,純 MC 變異可能已可接受)?可 A/B。
- 五張型 void 的精確邏輯(組合層級)。
