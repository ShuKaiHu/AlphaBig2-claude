# Big 2 PPO experiment

Independent PPO training that **reuses the existing rules/game engine**, built to
follow Patwa, *Self-Play RL under Imperfect Information in Big 2*
(arXiv 2605.28863). Lives in the `ppo-experiment` git worktree with its own
`.venv`; the original AlphaZero/MCTS project on `main` is never modified.

## What follows the paper

- **PPO** (it beats value-based methods in the paper's limited-compute setting).
- **Current-policy self-play**: all 4 seats use the current policy; we train on
  every seat. The paper shows this beats checkpoint self-play and fixed-opponent
  (Smart-only) training — even when *evaluating* against Smart. The heuristics
  are the yardstick, not the sparring partner.
- **Moderate entropy** (β = 0.05) to keep the policy from collapsing.
- **Hyper-params**: 4 epochs, clip 0.2, γ 0.99, λ 0.95, value coef 0.5,
  grad-norm 0.5, minibatch 256, 64 games/batch, lr 3e-5 with warmup + cosine.
- **Eval** vs fixed Random / Greedy (Alg 1) / Smart (Alg 2) pools, 1000
  seat-randomized games. Success = win-rate > 25% **and** avg-score > 0.

## Two architectures (A/B)

| `--arch` | state | action scoring | params |
|----------|-------|----------------|--------|
| `mlp` | 412-bit flat vector (`getCurrentState`) | fixed 14739-way masked head | ~8.0M |
| `cardaware` | card embeddings + hand self-attention | dot-product over **legal** actions only | ~0.1M |

`cardaware` is the paper's design. It matters here because ~70% of our 14739
actions are suit-variants of straights (`4♦5♦6♦7♦8♦` vs `…8♣` …): a fixed head
treats them as unrelated output slots, while dot-product over 64-dim action
features makes near-identical actions share learning. (See the chat discussion
for the full rationale.)

## Run (from the worktree root, so `actionIndices.pkl` resolves)

```bash
source .venv/bin/activate
python -m ppo.ppo_trainer --arch cardaware --updates 200      # paper design
python -m ppo.ppo_trainer --arch mlp       --updates 200      # baseline
```

Checkpoints → `ppo/checkpoints/ppo_<arch>_*.pt` (gitignored). The paper's budget
was 5000 batches × 64 games ≈ 320k games; scale `--updates` toward that for a
real model.

## Files

- `ppo_trainer.py` — self-play collection, per-seat GAE, PPO update, eval, LR schedule.
- `policies.py` — uniform interface (`act/greedy/build_minibatch/evaluate`) over both archs.
- `network.py` — `ActorCritic` MLP (412-bit + 14739 head).
- `network_cardaware.py` — card embedding + self-attention + dot-product scorer.
- `action_features.py` — precomputed (14739 × 64) per-action feature table (cached to disk).
- `eval_baselines.py` — Random / Greedy / Smart opponents + `evaluate`.

## Deviations from the paper (deliberate)

- **Reward = our engine's *multiplied* Big 2 score** (bombs / 2s / hand-size
  doubling), because that matches real online scoring — the actual objective
  (beating online humans on avg-score). The PPO learning signal is divided by
  `--reward-scale` (default 13) for stability; eval reports the TRUE score. As a
  result our avg-score magnitudes are NOT comparable to the paper's unshaped-score
  numbers, but the success criterion (win>25% & score>0) still applies.
- **No plain flush** — this engine removed it; Smart's BreakPenalty skips the
  flush term. Note the paper's game *has* flush, so cross-paper strategy differs.
- Smart's "low orphan" threshold and "very strong trick" are documented judgment
  calls (the paper leaves them informal).

## A/B results — 400-update run (main result)

400 updates × 48 games/batch (~19k games each); eval every 20 updates × 400
seat-randomized games. Reward is our multiplied score (larger magnitudes than the
paper). `best.pt` is kept by best Smart avg_score. avg_score is the headline metric.

**Best checkpoint per arch (ranked by Smart avg_score):**

| arch | best @ upd | vs Random | vs Greedy | vs Smart |
|---|---|---|---|---|
| `mlp` (8.0M)       | 100 | 67.5% / +11.6 | 38.8% / +3.3 | 25.2% / **+0.58** |
| `cardaware` (0.1M) | 120 | 73.2% / +16.9 | 33.2% / +3.6 | 24.0% / **+3.04** |

**Read (the 400-update picture is decisive, unlike the 40-update trend):**
- Both crush Random and clear Greedy (win>25% & score>0 by ~upd40–60).
- **Smart is the discriminator, and cardaware clearly wins it.** From ~upd60 on,
  cardaware's avg_score vs Smart is **consistently positive (+1 to +3)** and it
  hits the paper's full success bar (win>25% AND score>0) vs Smart at several
  points (e.g. upd180 27%/+1.4, upd300 28%/+3.0, upd400 26%/+1.9).
- **mlp never really beats Smart**: its Smart avg_score oscillates around 0 to
  −1.5 the whole run (best a marginal +0.58), win-rate stuck ~20–25% (≈ chance
  for a 4-player seat).
- cardaware also dominates on the easier opponents (Greedy +4…+7.6 vs mlp +2…+4;
  Random +16…+20 vs mlp +11…+16) — **with 76× fewer params** (0.1M vs 8.0M).
- Takeaway: the card-aware / dot-product architecture extracts much more from the
  same data, exactly as predicted for our straight-by-suit action explosion.

Caveats: ~19k games is still far below the paper's ~320k; eval = 400 games so
±a few % noise remains; both are vs rule-based bots, not humans (see
`SURVEY_evaluation.md`). Next: push cardaware further and then collect real-human
games.

<details><summary>Earlier 40-update trend check (superseded)</summary>

Both learned; cardaware started slower (embeddings from scratch) but pulled ahead
on Smart by upd30. The 400-update run above confirms and sharpens this.
</details>

## cardaware 1500-update long run — it plateaus

1500 updates (~72k games, ~5h43m), eval every 50 × 500 games. Best by Smart
avg_score = **upd 950**: Random 73.4% / +17.6, Greedy 38.0% / +4.8,
**Smart 27.8% / +3.28** → `ppo_cardaware_best.pt`.

- **Plateau**: vs Smart, avg_score climbs to ~+2…+3 by ~upd150–500 and then just
  oscillates there for the next ~1000 updates (best +3.28 @ upd950, end +1.33 @
  upd1500). vs Greedy plateaus ~+5; vs Random ~+18.
- **Diminishing returns**: 3.75× more training than the 400-run (+3.04→+3.28 vs
  Smart, win 24%→28%) — small. **Pure current-policy self-play has roughly hit
  its ceiling against the fixed Smart bot.**
- **Win-rate nuance**: vs Smart, win-rate sits ~22–28% (25% = chance for a
  4-player seat). The positive avg_score comes from *losing less / winning by
  bigger margins*, not from winning far more often.
- Prior best preserved as `run400_*.pt`.

**Implication**: more of the same self-play won't help much. To get stronger,
change the lever — richer architecture/features or inference-time search, or
go to the real objective (humans). Smart is a weak no-lookahead bot; we've
near-maxed the offline filter, so real-human games (see `SURVEY_evaluation.md`)
are the next meaningful signal.
