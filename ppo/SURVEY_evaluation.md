# Survey: how Big 2 (and adjacent card-game) papers evaluate model STRENGTH

Focus = **evaluation methodology only** (opponents, metric, #games, human testing),
not the training method. Built for this project's goal: a model that beats real
online human Big 2 players, judged by avg-score. Compiled 2026-06-19.

Metric glossary:
- **avg score / ADP** — average game points (winner gets Σ of losers' remaining
  cards; losers get −own remaining). "Average Difference in Points" (ADP) in the
  DouDizhu/GuanDan line is the same idea. This is the metric closest to our goal.
- **WP** — winning percentage (fraction of games won).
- **Elo / TrueSkill** — relative rating from many pairwise games.
- **seat-randomized** — agent's seat rotated across games so position (e.g. who
  holds 3♦ / acts first) doesn't bias results.

---

## Big 2 — the directly relevant papers

### 1. Charlesworth 2018 — *Application of Self-Play RL to a Four-Player Game of Imperfect Information* (this project's upstream)
arXiv:1808.10442 · https://arxiv.org/abs/1808.10442 · code: https://github.com/henrycharlesworth/big2_PPOalgorithm
- **Opponents**: (a) 3 × random, (b) **earlier versions of itself** (self-play progress), (c) **humans**.
- **Metric**: average score per game (native Big 2 scoring) + games-won count.
- **#games**: 10,000 vs random; 10,000 vs earlier-self; **664 vs humans** (7 players, 31–250 games each).
- **Human eval**: 7 amateurs ("none experts, all had some experience"); humans
  averaged **−0.96 ± 0.38** → the net "significantly outperforms most human
  players." Hosted a public web app for play.
- **Verdict**: the most concrete Big 2 human eval that exists — but only
  **amateur** opponents, modest sample. Strong opponents / experts untested.
- **Borrow**: avg-score-vs-random curve over training; vs-earlier-self as a
  self-play progress signal; a **public webapp to crowd-collect human games**.

### 2. Patwa 2026 — *Self-Play RL under Imperfect Information in Big 2* (the paper we follow)
arXiv:2605.28863 · https://arxiv.org/abs/2605.28863
- **Opponents**: 3 fixed heuristics — Random, Greedy (weakest legal play), Smart
  (rule-based scoring). **No humans, no cross-play.**
- **Metric**: win rate + avg score, per opponent class.
- **#games**: 1,000 seat-randomized per eval; single seed (noted as a limitation).
- **Success criterion**: win rate > 25% **and** avg score > 0.
- **Verdict**: clean, reproducible, but entirely "filter-level" — Smart (a
  no-lookahead bot) is the strongest bar; says nothing about human-level play.
- **Borrow**: the Greedy/Smart yardstick + success criterion + seat
  randomization (we already use all three in `eval_baselines.py`).

### 3. Big2AI 2022 — *Challenging AI with Multi-Opponent & Multi-Movement Prediction for Big2*
IEEE Access · https://ieeexplore.ieee.org/document/9755938/
- **Method**: MCTS / ISMCTS + opponent move prediction (search-based, not pure RL).
- **Opponents**: other computer AIs **and human players**.
- **Metric**: highest win rate + **least losing points** vs both.
- **Eval setup**: an **Android 4-player prototype** used to run the human games.
- **Verdict**: explicitly claims beating computer + human opponents; but sample
  size / human skill level not clearly documented (read from abstract).
- **Borrow**: "least losing points" framing; an app as the human-eval vehicle.

### 4. MDP-based Big2 2024 — *MDP-Based AI with Card-Playing Strategy & Free-Playing Right Exploration*
https://www.researchgate.net/publication/382093419
- **Opponents**: prior Big 2 AI baselines. **Human testing: not evident** (abstract-level only).
- **Metric**: win rate / points vs baselines.

### 5. Deep Monte-Carlo Big2 2024 — *Improved learning efficiency of deep MC for complex imperfect-info card games*
ScienceDirect · https://www.sciencedirect.com/science/article/abs/pii/S1568494624003193
- **Opponents**: state-of-the-art Big 2 AI benchmarks (AI-vs-AI). **No human eval evident.**
- **Metric**: win rate / score vs SOTA; emphasis on sample efficiency.

**Big 2 bottom line**: only Charlesworth (amateurs, 664 games) and Big2AI 2022
(app, under-documented) touch humans. **No rigorous expert/strong-human Big 2
benchmark exists** — a genuine gap our online-vs-human plan could fill.

---

## Adjacent card games — where human/strong eval is more standard (reference)

### DouZero / DouZero+ (DouDizhu / 鬥地主)
- **Metrics**: **WP** and **ADP** over large tournaments vs strong baselines.
- **Strength proof**: DouZero+ **ranked #1 on the Botzone leaderboard among 400+
  agents** (AI-vs-AI competitive ranking, not direct human).
- https://www.researchgate.net/publication/363723177

### DanZero / DanZero+ (GuanDan / 掼蛋)
arXiv:2210.17087 · https://arxiv.org/pdf/2210.17087
- **vs AI**: WP/ADP vs rule-based + RL baselines, large match counts.
- **vs humans**: **10 proficient grad students (not pros), 2 humans vs 2 AI, 20
  rounds** → "human-level." Small, non-expert sample.

### Gold-standard human evals in other imperfect-info games (what "rigorous" looks like)
- **Suphx (Mahjong)** — played on Tenhou vs thousands of human players, reached
  stable 10-dan; large-sample, ranked-ladder eval.
- **Pluribus (6-player poker)** — vs elite human professionals, using **AIVAT**
  variance reduction to get significance from limited hands.
- **Liar's Poker 2025** — "beats elite humans." arXiv:2511.03724 · https://arxiv.org/pdf/2511.03724

---

## Takeaways for THIS project

1. **Our offline yardstick already matches the field** (Random/Greedy/Smart +
   win-rate + avg-score + seat-randomization). Good for iteration; not a strength
   claim.
2. **The discriminating metric is avg-score (ADP)**, which is also our literal
   goal — keep ranking checkpoints by it (we now save `best.pt` by Smart score).
3. **For a credible human result, beat the Big 2 literature on rigor**:
   - Many games (Big 2 single-game variance is huge — Charlesworth used 664 vs
     humans, 10k vs bots; aim high, not the DanZero-style 20).
   - State opponent strength honestly (online ranked players ≠ "amateur friends").
   - Consider variance reduction (duplicate/mirror deals like AIVAT in poker) so
     fewer human games still give significance.
   - A small web/app harness (à la Charlesworth's webapp, Big2AI's Android app)
     is the practical way to collect real-human games.
4. **There is no strong-human Big 2 benchmark** — landing one would be a
   contribution, not just a check.
