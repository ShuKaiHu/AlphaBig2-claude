# KL divergence 在本專案的角色 — 完整介紹與論文地圖

> 對應 code:`ppo/ppo_trainer.py`(KL 錨定 + PPO 診斷 KL)、`ppo/train_v6_rl.py`(診斷 KL)。
> 對應戰略:M3「錨定 RL 機制探針」(STATUS.md 線 2/2b),KL 錨 = 凍結的 `policy_4500`。

---

## 1. KL divergence 是什麼

Kullback–Leibler divergence(相對熵)衡量「用分布 q 去近似分布 p 時,平均每個樣本多付出的資訊量」:

```
KL(p ‖ q) = E_{x~p}[ log p(x) − log q(x) ]
```

四個關鍵性質:

1. **非負**,且 `KL = 0 ⇔ p = q`(所以適合當「偏離程度」的量尺)。
2. **不對稱**:`KL(p‖q) ≠ KL(q‖p)`,方向的選擇有實質行為意義(見 §3)。
3. **單位是 nats**(用自然對數)。直覺換算:KL ≈ 0.02 nats 表示兩個分布幾乎重合——典型動作的機率比大約只差幾個百分點;0.7 nats 才接近「機率差一倍」的量級。
4. **不是距離**(不對稱、不滿足三角不等式),但局部(p ≈ q 時)二階展開是 Fisher information 度量,這是 natural gradient 與 trust region 方法的數學根基。

出處:Kullback & Leibler (1951), *On Information and Sufficiency*, Annals of Mathematical Statistics 22(1):79–86。

---

## 2. 本 repo 實際用到 KL 的三個地方

### 2a. KL 錨定(anchored RL)— 主角

`ppo/ppo_trainer.py:271-276`:

```python
with torch.no_grad():
    logp_ref, _, _ = kl_ref.evaluate(batch)   # 凍結的 policy_4500
dr = logp_ref - logp                          # grad 走 logp
kl_ref_term = (torch.exp(dr) - 1.0 - dr).mean()
loss = loss + kl_beta * kl_ref_term
```

- 目標函數是 **`reward − β · KL(π ‖ π_ref)`**,`π_ref` 是**凍結**的 BC policy(policy_4500),`β = kl_beta = 0.02`。
- 用意:讓 RL 只能在「BC 先驗的可信流形」附近搜索——policy 可以為了分數改變決策,但不能漂移成 BC 沒見過的怪異打法。這正是 M3 設計成「機制探針」的原因:RL1/RL2 兩次 null 的前科顯示,無錨 RL 在這個環境裡會學崩或學不到;錨定後「有沒有動囤2率」才是乾淨的因果問題。
- 這個配方和 RLHF(語言模型對 SFT policy 錨定)、AlphaStar(對人類模仿 policy 錨定)是**同一個家族**,見 §4。

### 2b. k3 估計器 — 那行 `exp(dr) − 1 − dr` 的來歷

KL 是期望值,minibatch 上只能抽樣估計。Schulman 的 blog 提出三個 single-sample 估計器(`d = p_ref/p`,樣本抽自 `p`):

| 估計器 | 公式 | 性質 |
|---|---|---|
| k1 | `−log d` | 無偏、高變異、**可為負**(KL 本身非負,樣本卻一半是負的) |
| k2 | `(log d)²/2` | 有偏、低變異、非負 |
| k3 | `(d−1) − log d` | **無偏(作為值)、低變異、非負** |

本 repo 用 **k3** 當可微 loss(`exp(dr)−1−dr` 就是 `(d−1)−log d`),和 GRPO(DeepSeekMath)的選擇相同。兩個實務注意事項:

- k3 作為 **loss 的梯度**並不是真 KL 梯度的無偏估計(近年文獻有專文分析,見 §4 末),但它本身是合法的 divergence(非負、π=π_ref 時恰為 0),且在 π ≈ π_ref 時與真 KL 一階等價——M3 全程 KL 只有 0.02–0.05 nats,在這個範圍差異可忽略。
- k3 在 `d` 很大時(policy 已嚴重偏離錨)會爆炸——這其實是 feature(強力拉回),但意味著 β 過小放任漂移後才開錨會不穩定。本 repo 從第 0 步就開錨,不踩這個坑。

### 2c. PPO 的診斷 KL(old vs new)

`ppo/ppo_trainer.py:285` 與 `ppo/train_v6_rl.py:113`:

```python
acc["kl"] += (logp_old_t[mb_t] - logp).mean().item()
```

這是 **k1 估計器**估 `KL(π_old ‖ π_new)`——每輪 update 內 policy 走了多遠。它只做**監控**不進 loss(進 loss 的約束由 PPO 的 ratio clip 承擔);log 裡 `kl` 欄印成 `+.4f` 帶正負號,就是因為 k1 樣本可為負。Spinning Up 等實作常用它做 early-stop(approx KL 超標就停該輪 epoch),本 repo 用固定 epochs + clip,把它留作健康指標。

### 2d. 隱藏的第三個 KL:entropy bonus

`− ent_coef · entropy` 這項(`ppo_trainer.py:270`)在數學上等價於對 **uniform 分布**的 KL 懲罰(差一個常數):最大化 entropy = 最小化 `KL(π ‖ uniform)`。所以整個 loss 其實同時掛著兩個方向不同的 KL 錨——一個拉向 BC 先驗(β=0.02),一個拉向均勻分布(ent=0.01),前者管「像不像人」,後者管「別坍縮成 argmax」。這兩個係數的拔河正對應 STATUS.md 記載的根因之一「argmax 坍縮」。

### 2e. 已排定的 TODO:adaptive β(target-KL controller)

`ppo_trainer.py:248` 註記 Phase 2 要做的 adaptive β,出處就是 PPO 論文的第二個變體:設定目標 KL(如 0.01 nats),實測 KL 高於目標 1.5 倍就把 β 加倍、低於 1/1.5 就砍半。Phase 1 固定 β 已夠用(錨全程 0.02–0.05 nats,又緊又沒卡死)。

---

## 3. 方向為什麼是 KL(π ‖ π_ref) 而不是反過來

- **Reverse KL(π‖π_ref,本 repo 用法)是 mode-seeking**:π 在 π_ref 機率≈0 的地方放質量會被重罰(`d→∞`),但允許 π 集中在 π_ref 眾多合理動作的**其中一個** mode 上。翻成牌桌語言:不准發明 BC 沒見過的怪招,但允許在 BC 覺得都合理的幾手牌之間,按分數重新分配偏好——這正是「錨定微調」要的。
- **Forward KL(π_ref‖π)是 mass-covering**:會逼 π 給 π_ref 的每個動作都留機率,行為上更像蒸餾/模仿(BC 的 cross-entropy loss 本身就是 forward KL)。拿來當 RL 錨會阻止 policy 收斂到少數好動作。
- 順帶:PPO ratio `exp(logp − logp_old)` 的 clip 也可視為對 reverse KL trust region 的粗糙一階代理(TRPO→PPO 的簡化路線,見 §4)。

---

## 4. 論文地圖(按概念層)

### 起源與教科書層
- **Kullback & Leibler (1951)** — *On Information and Sufficiency*. KL 的原始定義(假設檢定的資訊量觀點)。
- Cover & Thomas, *Elements of Information Theory* — ch.2 對 KL 性質(非負、鏈式法則、與互資訊關係)最完整的標準參考。

### Trust region 層(KL 當「步長尺」)
- **TRPO** — Schulman et al. (2015), *Trust Region Policy Optimization*, [arXiv:1502.05477](https://arxiv.org/abs/1502.05477)。單調改進理論 + 「每步 KL(π_old‖π_new) ≤ δ」的約束優化;是 §2c 診斷 KL 的理論源頭。
- **PPO** — Schulman et al. (2017), *Proximal Policy Optimization Algorithms*, [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)。兩個變體:clip(本 repo 的 `pi_loss`)與 **adaptive KL penalty**(§2e 的 TODO 即此)。
- **MPO** — Abdolmaleki et al. (2018), *Maximum a Posteriori Policy Optimisation*, [arXiv:1806.06920](https://arxiv.org/abs/1806.06920)。把 trust region 寫成 EM 形式的 KL 約束,trust-region-as-KL 的另一條成熟路線。

### KL 正則化 RL 的理論層(KL 當「錨」為什麼有效)
- **Vieillard et al. (2020)** — *Leverage the Average: an Analysis of KL Regularization in RL*, [arXiv:2003.14089](https://arxiv.org/abs/2003.14089), NeurIPS 2020。**理解「錨為什麼幫忙」的首選理論文**:證明 KL 正則化隱式地對歷代 Q 值取平均,把誤差累積變成誤差平均,performance bound 從 horizon 二次依賴降到線性。
- **Levine (2018)** — *Reinforcement Learning and Control as Probabilistic Inference*, [arXiv:1805.00909](https://arxiv.org/abs/1805.00909)。RL-as-inference 統一視角:soft/max-entropy RL 就是對某個先驗的 KL 正則化;把 §2d(entropy=對 uniform 的 KL)講透的 tutorial。
- Haarnoja et al. (2018) — *Soft Actor-Critic*, [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)。max-entropy RL 的代表實作(對 uniform 錨那一支)。

### Anchored RL / behavior prior 層(和 M3 同款的做法)
- **Distral** — Teh et al. (2017), *Distral: Robust Multitask Reinforcement Learning*, [arXiv:1707.04175](https://arxiv.org/abs/1707.04175)。多任務各 policy 對一個共享蒸餾 policy 做 KL 錨定;「reward − β·KL(π‖π_0)」這個目標式的經典出處之一。
- **Galashov et al. (2019)** — *Information Asymmetry in KL-regularized RL*, [arXiv:1905.01240](https://arxiv.org/abs/1905.01240), ICLR 2019。系統性研究「錨 policy 該看到什麼資訊」:給 π_ref 較少資訊會逼它學到可重用的 default behavior。對「錨要不要跟主 policy 同架構同輸入」這類設計題最有參考價值。
- **Jaques et al. (2019)** — *Way Off-Policy Batch Deep RL of Implicit Human Preferences in Dialog*, [arXiv:1907.00456](https://arxiv.org/abs/1907.00456)。KL-control 用於錨住 pretrained 對話模型;前作 Sequence Tutor([arXiv:1611.02796](https://arxiv.org/abs/1611.02796))是「RL 微調 + KL 拉住 supervised 先驗」最早的清楚示範之一。
- **AlphaStar** — Vinyals et al. (2019), *Grandmaster level in StarCraft II using multi-agent reinforcement learning*, Nature 575:350–354。RL 全程對人類模仿 policy 加 KL cost,防止漂出人類策略流形——與 M3「錨=BC policy_4500」動機完全同構(探索空間巨大、無錨 RL 會漂進自嗨策略)。

### RLHF 層(目前最大規模的 KL 錨應用)
- **Ziegler et al. (2019)** — *Fine-Tuning Language Models from Human Preferences*, [arXiv:1909.08593](https://arxiv.org/abs/1909.08593)。`reward − β·KL(π‖π_SFT)` 進 PPO 的定式化;RLHF 的 KL 錨從這裡定型。
- Stiennon et al. (2020) — *Learning to Summarize from Human Feedback*, [arXiv:2009.01325](https://arxiv.org/abs/2009.01325);Ouyang et al. (2022) — *InstructGPT*, [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)。同配方 scale 化;附帶大量「β 太小→reward hacking、β 太大→學不動」的實務觀察,對調 `kl_beta` 直接可借鑑。
- **DPO** — Rafailov et al. (2023), [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)。證明 KL 錨定 RLHF 目標有閉式解,可化成純監督 loss——想深入理解「β 在數學上到底控制什麼」,DPO 的推導最乾淨。
- **GRPO** — Shao et al. (2024), *DeepSeekMath*, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)。把 k3 估計器直接當可微 KL loss——與本 repo `ppo_update` 的寫法相同。

### KL 估計器層(§2b 那行 code 的文獻)
- **Schulman (2020)** — *Approximating KL Divergence*, [joschu.net/blog/kl-approx.html](http://joschu.net/blog/kl-approx.html)。k1/k2/k3 的出處;本 repo docstring 說的「k3 estimator」即此。
- **後續分析(選讀)** — *On a Few Pitfalls in KL Divergence Gradient Estimation for RL* ([arXiv:2506.09477](https://arxiv.org/abs/2506.09477))等近作:k3 當 loss 的梯度不等於真 KL 梯度、各估計器「值無偏 vs 梯度正確」的取捨。錨很緊(<0.1 nats)時實務差異不大,但 Phase 2 若要上 adaptive β、或觀察到 klref 曲線異常,值得回來讀。

---

## 5. 建議閱讀順序(針對本專案)

1. **Schulman blog(kl-approx)** — 10 分鐘,直接對上 `ppo_update` 的每一行。
2. **PPO 論文 §4** — clip 與 adaptive KL penalty 兩變體,涵蓋 §2c 與 §2e。
3. **Ziegler 2019 + AlphaStar** — 「凍結先驗 + β·KL 錨」在兩個不同領域的成功先例,理解 M3 設計的定位。
4. **Vieillard 2020** — 想知道「錨為什麼在理論上就該有效」再讀這篇。
5. **Galashov 2019 / Distral** — Phase 2 若考慮改錨的架構或資訊輸入時讀。

## 6. 與本專案鐵律的對接備忘

- KL 曲線(`klref` 欄)是**事前判讀規則的一部分**(STATUS.md 線 2:5–8pp 灰帶要查 KL/entropy 曲線再裁)——它是機制證據,不是可以事後挑著看的記帳。
- 「錨全程 0.02–0.05 nats」的意義:policy 從未離開 BC 流形,所以 D2-via-RL 的 NULL 是「在錨內學不到」的乾淨結論,而不是「policy 崩了」的髒結論。β 掃描若未來立案,判讀規則要在跑之前寫死。
