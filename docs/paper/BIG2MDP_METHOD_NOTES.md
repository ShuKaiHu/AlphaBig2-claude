# Big2MDP(ToG 2025)方法研讀筆記

> 來源:Chen & Lu, IEEE ToG 17(2):267–281 全文(2026-08-12 精讀)。
> 用途:(1) related work 的技術描述正確性;(2) 可借鏡元件清單(§5);(3) 潛在 baseline 復刻規格(§6)。

## 1. 地基:狀態抽象 + 頻率轉移表(無神經網路)

- **狀態 = 一個回合**:s = {(p0,a0),(p1,a1),(p2,a2),(p3,a3)}(四家該回合的動作);
  決策推演時自己的動作以萬用符 X 抽象(共享狀態、壓縮狀態數)〔式 (1)(2)〕。
- **轉移機率 = 計數**:Pa(s,s′)=N(s,a,s′)/N(s,s′)〔式 (3)〕,從 **50 萬局 AI 互打**累積(無人類資料)。
- 動作 = 手上可組出的合法牌組(枚舉)或 Pass;跟牌時不合法動作不進動作集。
- ⚠️ 技術細節:他們的 Q **不是期望值**——Q1.0 = max_si [P(s,si)×R(s,si)] 是「逐路線 P×R 取 max」的
  樂觀啟發式(負分路線直接忽略,Fig 2 範例明示),不是教科書 MDP 的 value iteration。復刻時別寫成期望。

## 2. 四個 reward 頭(逐代補洞)

| 版本 | 目標 | 核心式 | 補的洞 |
|---|---|---|---|
| MDP 1.0 | 搶分 | Q1.0 = max P_win×R_win〔(4)-(6)〕 | —(賭徒:只看大勝路線) |
| MDP 2.0 | 止損 | Q2.0 = max[P_win×R_win + P_lose×R_lose]〔(7)〕+ **Send 局末開關**〔(8)(9)〕 | 1.0 無視輸的代價 |
| MDP 3.0 | 速勝 | 權重 W(s′)=1−d(s′)/dmax,d=到勝利的最短狀態數〔(10)-(13)〕 | 2.0 會拖局 |
| MDP 4.0 | 勝率 | Qwin = max P_win(拔掉分數/速度)、Qlose = max P_lose×R_lose〔(14)-(16)〕+ 三策略(§3) | 高分路線易被攔截 |

**Send(局末模式開關)**〔式 (8),調參後 Eα=4, Eβ=4, Eγ=30〕:
任一成立即進入「局末」——(a) 任一對手剩牌 ≤4;(b) 我的**非單張**牌組數 ≤4;(c) 檯面已出牌總值等級 ≥30。
局末且所有勝路 Q≤0 → 切到止損頭:**寧可先丟 2 保住倍數**(Fig 3 範例:剩 d9+c2,對手快走,
先丟 Club 2,輸時只賠 5 而不是 5×2)。⚠️ 這就是我們 G2 的 last-chance 2-dump 行為,
他們用一個 if-else 開關實現——對照我們 M3 Phase 1 的 D2-via-RL null,方法論對比極鮮明。

## 2.5 目標函數的心路歷程:為什麼從 max score 一路退到 max win rate(逐字引文)

大老二分數是 **winner-take-all**(1.0 的定義處就寫明:"In Big2, only the winner can earn
points, while the other three players lose points" — 因此連 1.0 都只計勝利劇本)。演進鏈:

| 轉折 | 病(論文自述) | 藥 |
|---|---|---|
| 1.0→2.0 | "weight the chance of a big win as worth risking the loss"(賭徒) | + 輸分項;為免全場保守再加 S_end 開關 |
| 2.0→3.0 | 分數 reward 不管誰先出完,但遊戲先出完即結束 | 離勝利距離加權 W |
| 3.0→4.0 | "any route can be intercepted";"the scoring rewards tend to make the MDP choose high-risk routes, which are more likely to be intercepted… may lead to even more points being lost";速度權重誘導囤大組 | **自白**:"we remove the rewards for the fastest winning and scoring, and solely use the probability of the action leading to a player's own victory state" |

**max win rate 如何換回 max score — 論文的分工(消融為證)**:
E[score] = P(win)×贏量 − P(lose)×輸量,三元件分治 —
P(win)←WP;輸量←S_end 止損頭;贏量←SP("This approach maximizes the opponent's score
reduction while increasing the winner's own score")。Fig 8 消融明示分工:**"Although SP has a
slight impact on win rates, it significantly affects the scores"**(WP/MP/WC 拉勝率、SP 拉分數);
Fig 11:純分數的 1.0 墊底,4.0 勝率與分數雙冠。思想源頭在 related work:Snakes & Ladders 的
「最短路 vs 最大勝場」對比 + 麻將多 MDP(單一 reward 過度偏重)。

**評註(我方分析,非論文)**:(a) 分數是無界量,配上他們 max-of-products 樂觀估計 + 頻率表噪音,
直接 max score 必被大 R 路線騙走;P(win) 有界 [0,1],對噪音穩健——這個選擇部分是配合自家估計器的
工程妥協。(b) 與本專案缺口分解(勝率通道占 77%)結構同構:兩條獨立路徑得到「大老二分數被勝率通道
主導、尾部靠止損」同一命題——paper discussion 可用的呼應。(c) 侷限:win-rate max 非 E[score] 完全
代理(小贏高勝率 vs 大贏低勝率),SP 是啟發式補丁非聯合優化;分解最優性從未被證明,只有經驗消融。

## 3. MDP 4.0 的三個策略疊層(優先序:SP > WC > win/loss)

- **MP 對手預測**〔(17)(18)〕:Crest = 全牌 − 已出 − 我手;從歷史狀態庫挑「特徵配對」的狀態建預測樹,
  四特徵 =(首出牌組、Table-Card Level、三家剩牌數、自家剩牌數),**OR 邏輯**(至少一個 match 即選入),
  再在配對子集上重新歸一轉移機率。= 粗粒度 case-based belief(遠弱於我們的 history belief net)。
- **WC 蓋爛牌**〔(19)-(24)〕:R_c(s,s′)=(K1+K2+K3)/3 = 三家全 pass 的比例 ≈ **這手拿住控權的機率**;
  Scover 三段條件:現在有一手拿權機率 ≥Eα → 下一步「恰好一手」爛牌 ≤Eβ → 再下一步又有一手 ≥Eγ
  → 進入「拿權→蓋爛牌→再拿權」三步規劃,狀態值改用 V_cover。
- **SP 連出收尾**〔(25)-(27)〕:Sseries = 手上 i 組牌中**至多 1 組**拿權機率低(≤Eβ)、其餘全部 ≥Eα
  → 直接連出到底(最後一手不需拿權),V_series = Σ R_c。**這是對 G1(強牌轉換)的顯式解**。

## 4. 工程面

- 每手決策平均耗時:MDP1.0 432ms / 2.0 741ms / 3.0 832ms / **4.0 2454ms**(Table II;i5-9500);
  自建 app 限時 10 秒出手 → 上線神來也的出手限時可行。
- 調參法:單變數掃描(100k 局訓練 + 1000 局對打評估);難度選單 Rookie/Normal/Expert/Master
  = MDP 1.0/2.0/3.0/4.0。C#/.NET server + MySQL + protobuf。

## 5. 對本專案:值得學 vs 不值得學

**值得學(每項都對映到我們既有元件)**:
1. **Send 開關 → 規則版 D2**:min_plays_to_empty(`value/hand_features.py`)+ 剩牌數就能做
   「局末且無勝路 → 先丟 2/貴牌」的 override。RL 學不動的行為(M3 null),規則一行就有。
   候選為 M4'(policy + 規則 override 混合臂),cheap 且機制精準。
2. **R_c 拿權機率 → 我們的 dominance + belief 是嚴格上位**:他們用頻率表估「會不會被壓」,
   我們的 dominance.py 是精確計算、belief net 是學來的分布——同一個量,更準的兩個實作。
3. **SP finisher check → G1 的顯式解**:牌組分解(greedy 已有)+ 每組 dominance/belief 估拿權率,
   「i−1 組拿得住 → 連出」;可做 policy override 或 search 的節點條件。
4. **回合級 macro-action 規劃**:狀態=回合、動作=牌組,深度大減——search 線降算力的借鏡。

**不值得學**:頻率轉移表(外星分布、粗)、MP 的 OR 特徵配對(我們 history belief 上位)、
手工單變數調參、max-of-products 的非期望 Q(數學上站不住,只是 heuristic 湊效)。

## 6. 「Big2MDPLite」復刻規格(若立案;需照 V0–V3 驗證梯 + 事前註冊)

policy 臂 + 兩個規則 override,無訓練、幾百行:
- (a) 牌組分解:`gameLogic.handsAvailable` + `min_plays_to_empty` greedy 分解為 i 組;
- (b) 每組估拿權機率:dominance(確定性)+ belief(機率化未見牌);
- (c) **SP check**:≥ i−1 組高拿權 → 進連出模式(對 G1);
- (d) **Send check**:對手剩牌 ≤4 ∨ 我的組數 ≤4,且無勝路 → 丟倍數最貴的險牌(對 G2);
- (e) 其餘照 policy。
定位:線上 A/B 的對照臂(「規則 override 能拿多少 G1/G2 分數」),也是 M3 null 之後的
機制歸因工具:若規則臂能動指紋而 RL 不能,病灶定位再添一證。
**紀律**:此為新方法臂,立案前需使用者同意 + 事前註冊判讀規則,不在本 session 擅自啟動。
