# Related Work 比較:規則變體 × 模型架構(2026-08-12 調查)

> 對照欄「本專案」= `OUR_RULES_AND_ARCH.md`(code 為準)。
> 證據等級:**[CODE]** = 直接讀原始碼/原始 PDF 驗證;**[SNIP]** = WebSearch 摘錄(頁面內容但未親抓);
> **[2nd]** = 引用文獻轉述;**[N/A]** = 全文拿不到,未知。
> 調查環境限制:sandbox proxy 封鎖 arxiv/IEEE/ScienceDirect/RG/Wikipedia/pagat 直接抓取;
> Charlesworth 靠 clone 其 GitHub repo 逐行驗證 [CODE];Patwa 拿到 arXiv PDF 鏡像
> (sha256 與獨立 crawler 記錄吻合,v1;**v2 存在但未讀**)[CODE];
> 兩篇台灣 Chen & Lu 論文與 Luo & Tan 全文皆不可達 [N/A] → **待辦:用有 IEEE/Elsevier 權限的管道補抓**。

## 0. 一句話結論

1. **規則**:「大老二」是一個規則家族,不是一個遊戲。五篇 Big2 論文沒有任何一篇用我們的變體;
   兩篇可完整驗證的(Charlesworth、Patwa)都用**自訂簡化規則**,彼此也不同。
   我們的 神來也/台灣變體組合(無三條無同花、繞順且 2-3-4-5-6 最大、炸彈越級、黑桃2 單張無敵、
   乘法結算、one-card rule)**在文獻中未出現過**。
2. **架構**:三軸(history 序列輸入 / self-attention / 雙塔動作打分)中,
   **雙塔 + 手牌 self-attention 是跟隨 Patwa 2026**(本 repo `ppo/README.md` 明載);
   **本專案的架構增量是 history GRU 塔**(196 步原始 required/played 編碼、保留 pass 事件)。
   就可查文獻範圍內,**沒有任何已發表卡牌 AI 同時具備三軸**;
   雙塔 dot-product 動作打分在卡牌遊戲中甚至只有 Patwa 一例(圈外前例:DRRN 文字遊戲、AlphaStar pointer)。

## 1. 規則變體比較(論文)

| 規則點 | **本專案(神來也)** [CODE] | Charlesworth 2018 [CODE] | Patwa 2026 v1 [CODE] | Chen & Lu 2022 (Big2AI) / 2025 (Big2MDP) | Luo & Tan 2024 (DMC) |
|---|---|---|---|---|---|
| 花色序(小→大) | **♣♦♥♠** | ♦♣♥♠ | ♦♣♥♠ | [N/A](台灣團隊,推測台規但未驗證) | [N/A] |
| 起手 | 持**3♣**,首手必含(任意含 3♣ 牌型) | 持 3♦;**引擎自動代打單張 3♦**(玩家無決策) | 持 3♦,首手必含 | [N/A] | [N/A] |
| 三條 standalone | **無** | 有 | 有 | [N/A] | [N/A] |
| 4 張牌型 | **無** | **兩對 + 無 kicker 鐵支**(4張牌型!) | 無 | [N/A] | [N/A] |
| 5 張牌型 | 順<葫蘆<鐵支+1<同花順;**無同花** | 順<**同花**<葫蘆<同花順;**無鐵支+1** | 順<同花<葫蘆<鐵支+1<同花順(全五種) | [N/A] | [N/A] |
| 順子窗 | **10 窗**:34567…10JQKA + A2345 + 23456;JQKA2 非法 | **9 窗**:34567…10JQKA + **JQKA2(最大)**;無繞順 | **未定義**(全文沒寫!) | [N/A] | [N/A] |
| 順子排序 | A2345 最小…10JQKA 次大,**23456 最大** | 頂牌點數比,JQKA2 最大 | 未定義 | [N/A] | [N/A] |
| 炸彈越級 | **鐵支/同花順壓任何張數** | **無**(嚴格同張數;鐵支只在 4 張 trick 內是小炸彈) | **無跨張數**(五張型內可越級:同花壓順子等) | [N/A] | [N/A] |
| 黑桃2 單張 | **無敵**(含鐵支/同花順都不能壓) | 只是最大單張,無特規 | 只是最大單張,無特規 | [N/A] | [N/A] |
| Pass | 自願;**pass 後本 trick 鎖定**;3 連 pass 清 trick | 自願;**不鎖定**(有人出牌即重置);3 連 pass 清 trick | 自願;無鎖定規則;3 連 pass 清 trick | [N/A] | [N/A] |
| One-card rule | **有**(下家剩 1 張→單張只准出最大) | 無 | 無 | [N/A] | [N/A] |
| 計分 | −剩牌數;**倍數**:每張2 ×2、鐵支 ×2、同花順 ×2、≥10張 ×2;贏家收尾炸彈 ×2、主 2 ×2(全桌乘) | 純剩牌數,**零倍數** | 純剩牌數,**零倍數** | 有「輸分」概念,公式 [N/A] | [N/A] |

註:
- Charlesworth 引擎 [CODE] 出處:`gameLogic.py`/`big2Game.py`/`rules.md`(clone 自 `henrycharlesworth/big2_PPOalgorithm`)。
  特異點:兩對是合法 4 張牌型、鐵支是 4 張牌型(無 kicker)、同花高於順子低於葫蘆、JQKA2 是最大順。
- Patwa v1 [CODE]:規則在 Sec 2 / App A;**順子窗全文未定義**——任何「follows Patwa」的規則宣稱都
  釘不住順子合法性。計分無倍數(value-based 算法訓練時 reward ÷13 只是數值縮放)。
- Chen & Lu 兩篇(IEEE Access 2022;IEEE ToG 2025,即 RG 382093419 那篇)與 Luo & Tan
  (Applied Soft Computing 2024,DOI 10.1016/j.asoc.2024.111545,DouDizhu+Big2 雙遊戲)
  規則細節全文不可達 [N/A]。Luo & Tan 的對手模型會預測到「花色級」手牌組成 [SNIP]。

## 2. 規則變體比較(地區;支持「規則家族」論述)

濃縮自 pagat / 中英文維基 / 港台平台說明頁(全部 [SNIP],proxy 擋直抓;來源列在文件尾)。

| 變異點 | 香港鋤大D | **台灣(神來也=本專案)** | 大陸锄大地 | 新加坡/馬來西亞 | 菲律賓 Pusoy Dos |
|---|---|---|---|---|---|
| 花色序 | ♦♣♥♠ | **♣♦♥♠** | ♦♣♥♠ | ♦♣♥♠ | **♣♠♥♦(方塊最大)** |
| 起手 | 3♦ | **3♣** | 3♦ | 3♦ | 3♣ |
| 三條/同花 | 有/有(三條有爭議) | **無/無** | 有/有 | 星馬多為無/無 | 有/有 |
| 2 入順 | 可(A2345/23456) | 23456 可;A2345 各家有爭議(神來也:合法且最小)| 可 | 可 | 可(繞順,A/2 當高牌) |
| 最大順 | A2345 或 23456(house rule) | **23456** | 同 HK | 23456 | 10JQKA 系最高、34567 絕對最小 |
| 炸彈越級 | 無(僅為 pagat 記載的變體) | **有(鐵支/同花順)** | 無 | 不一 | 無 |
| 計分 | 1/張;8+→2n、10+→3n、13→4n;抱♠2 再 ×2 | −1/張;≥10 ×2;每張2 ×2;(13張×3 為 pagat 記載變體,神來也未確認) | 同 HK | — | 依桌約 |
| One-card(頂大/報牌) | house rule | **平台強制** | house rule | — | 非正式 |

⚠️ 兩個要在 paper 前對平台再驗的點:
1. **黑桃2 單張無敵**:一般台灣規則文獻說鐵支/同花順「可攔單張黑桃2」;我們的引擎寫死「無敵」
   (`rules.md`、`returnAvailableActions`)。引擎是照平台實測修出來的,但建議翻線上 log 找
   「有人對單張♠2 出炸彈被拒/成功」的實例做最終確認,論文才敢寫死。
2. **輸家倍數的重疊規則**:神來也說明頁提到剩牌同時含鐵支與同花順時「僅計算鐵支」[SNIP];
   我們 `_hand_multiplier` 是兩者相乘(×4)。牌型重疊是極罕見 edge case,但值得對一次帳
   (`_winner_finish_multiplier` 的 4412/4412 驗證不覆蓋這個函式)。

## 3. 架構三軸比較

三軸:(i) **history 序列輸入**;(ii) **transformer/self-attention**;(iii) **動作表徵**(固定頭 vs 動作當輸入,及打分方式)。

| 系統 | (i) History | (ii) Attention | (iii) 動作表徵 | 證據 |
|---|---|---|---|---|
| **本專案** | **✅ GRU(128) over 196 步**,每步 = 座位 + 當步「須回應的牌」+「實際出的牌」raw 52-hot;**pass 事件保留** | 手牌:**單層 4-head self-attention**(集合編碼;非完整 transformer) | **雙塔**:state 塔 × 動作塔(64 維特徵→MLP),縮放內積,**只打合法動作** | [CODE] 本 repo |
| Charlesworth 2018 | ❌ 僅累積指示器(高牌已出、對手出過的牌型 flag);**無序列** | ❌ 純 MLP(412→512→256) | **固定 1695-way 頭** + −inf mask | [CODE] repo |
| Patwa 2026 | ❌ 聚合指示器(seen/per-opp played 52-hot);**無序列、無 RNN** | ✅ **手牌 masked self-attention**(層數/頭數 v1 未載) | ✅ **雙塔 dot-product**:80 維動作特徵→2 層 MLP,線性投影 + 縮放內積,只打合法動作 | [CODE] PDF v1 |
| Chen & Lu 2022/2025 | 非深度學習(MCTS/ISMCTS + 啟發式;MDP + 規則) | — | — | [SNIP] 摘要 |
| Luo & Tan 2024 | DMC(DouZero 系)推測有 LSTM,**未驗證** | 無記載 | DMC 系 = 動作當輸入 Q(s,a)(concat 式,推測) | [SNIP]/[2nd] |
| DouZero | ✅ **LSTM over 最近 15 手**(5×162) | ❌ | 動作當輸入,**concat + 6 層 MLP** → Q;非 dot-product | [CODE] repo |
| DanZero / + | 扁平特徵(RNN 有無未驗證);+ 版改 PPO 仍動作當輸入 | 無記載 | concat 式 Q(s,a) | [SNIP] |
| PerfectDou | 推理路徑無 RNN/attention | ❌ | 合法動作全部 concat 進輸入 → logits;非 dot-product | [CODE] repo |
| Suphx | ❌ 歷史壓成通道平面(GRU 只在全局 reward 預測器) | ❌(ResNet CNN) | 固定離散頭(多模型) | [SNIP] |
| AlphaHoldem | 歷史壓成張量(CNN);**狀態側偽孿生雙塔**(牌/下注),非動作打分雙塔 | ❌ | 固定(抽象)動作頭 | [SNIP] |
| Tjong(麻將 2024) | — | ✅ 真 transformer(監督學習,非 RL) | 兩階段離散決策 | [SNIP] |
| DRRN 2016(文字遊戲) | — | — | **state/action 雙塔內積 Q** —— dot-product 雙塔的圈外正主前例 | [SNIP] |
| AlphaStar | LSTM core | ✅ entity transformer | pointer network 對變動候選集打分(attention 式) | [SNIP] |

### 誠實定位(paper 的 novelty 句怎麼寫)

1. **不是我們的**:手牌 self-attention、合法動作雙塔 dot-product —— 均出自 Patwa 2026(應明引;
   本 repo `ppo/README.md` 本來就寫「paper architecture」)。與 Patwa 的小差異:動作特徵 64 維
   (他 80 維)、state MLP+LayerNorm(他 projection+LN+residual FF)。
2. **是我們的(架構面)**:**history GRU 塔** —— 整局(≤196 步)原始 required/played 逐步編碼,
   **顯式保留 pass 事件**(「拒絕壓 X」= belief 的持久約束訊號,聚合指示器表達不了)。
   DouZero 的 15 手 LSTM 是最近似前例,但短窗、無 required-combo 語境、concat 非雙塔。
3. **組合宣稱**(可查範圍內):沒有已發表卡牌 agent 同時具備「RNN 全局歷史 + 手牌 attention +
   合法動作雙塔 dot-product」;卡牌圈的 dot-product 動作塔只有 Patwa(2026-05 preprint)。
   ⚠️ 措辭要保守:負向宣稱基於 search 級覆蓋(proxy 擋全文庫),且 Patwa v2 未讀。
4. **「transformer」一詞不要用**:我們的是單層 `nn.MultiheadAttention` 集合編碼,無 FFN/殘差堆疊/
   位置編碼。建議寫 "a single multi-head self-attention layer over card embeddings (set encoder)"。
5. 訓練面的差異才是主戰場:Patwa/Charlesworth 均為純 self-play、對 bot 評估;我們是
   真人資料 BC → AW-BC → KL 錨定 PPO,線上對真人 A/B(見 `PAPER_PLAN.md` 貢獻 2/3)。

## 4. 待辦(投稿前補洞)

- [ ] 取得 Patwa **v2** 全文,diff v1(尤其架構細節與是否補了順子定義)。
- [ ] 用有權限的管道抓 Chen & Lu 2022(IEEE Access 開放取用,校內網路可下)與 2025 ToG、
      Luo & Tan 2024 全文 → 補表 1 的 [N/A] 欄(特別是:台灣團隊用的是不是台規?有沒有同花?)。
- [ ] 線上 log 驗證:單張♠2 是否真的連炸彈都不能壓(§2 警示 1)。
- [ ] `_hand_multiplier` 鐵支/同花順重疊計分對帳(§2 警示 2)。
- [ ] Charlesworth 人類對戰數字(humans −0.96±0.38, 664 局)以 PDF 原文再確認正負號歸屬
      (現依 repo + 本 repo SURVEY 交叉讀為「人類均分 −0.96 → AI 為正」)。

## 5. 來源

- Charlesworth 2018:arXiv:1808.10442;repo `henrycharlesworth/big2_PPOalgorithm`(clone 驗證:
  `rules.md`、`gameLogic.py`、`big2Game.py`、`enumerateOptions.py`、`PPONetwork.py`、`mainBig2PPOSimulation.py`)。
- Patwa 2026:arXiv:2605.28863 **v1** PDF(GitHub 鏡像,sha256 `8030124f…7491b3` 與獨立 crawler 記錄吻合)。
- Chen & Lu 2022:IEEE Access 10, pp.40661–40676(doc 9755938);Chen & Lu 2025:IEEE ToG 17(2)
  pp.267–281, DOI 10.1109/TG.2024.3424431(= RG 382093419)。僅摘要級。
- Luo & Tan 2024:Applied Soft Computing, DOI 10.1016/j.asoc.2024.111545(DouDizhu + Big2)。僅摘要級。
- DouZero:arXiv:2106.06135 + repo `kwai/DouZero`(clone 驗證 models.py/env.py);
  PerfectDou:NeurIPS 2022 + repo(clone 驗證);DanZero(+):arXiv:2210.17087 / 2312.02561;
  Suphx:arXiv:2003.13590;AlphaHoldem:AAAI 2022;Tjong:CAAI TIT 2024;DRRN:arXiv:1511.04636。
- 地區規則:pagat.com Big Two;zh.wikipedia 鋤大弟;神來也 gamesofa 說明頁;戲谷/gm99/jbtjbt/
  mirowebs/stheadline/hk01/Baidu(細目見表 2;全部 [SNIP] 級)。
