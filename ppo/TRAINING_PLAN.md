# Training plan — beat real humans at Big 2

> **Audience:** future AI coding agents (and the human author) continuing this line.
> Read this first. It explains *what we are optimizing for*, *why the opponent pool
> is built the way it is*, and *the loop that makes it better over time*. Concrete
> file paths and commands are at the bottom (§9).

---

## 1. North star — and how we measure it

**Goal: win against real human players on 神來也大老二 (gamesofa), measured by
average score (`avg_score` / reward), NOT win-rate.** Big 2 is 4-player and scored
by multiplied card-count differentials, so "lose less / win bigger" matters more
than "win more hands." A model that wins 18% of hands but loses −3.8/hand is better
than one that wins 20% but loses −10/hand.

Offline metrics (vs Random/Greedy/Smart bots, BC val accuracy, belief P@count) are
**only filters**. The real verdict is **online vs humans**, and it needs ~50 rounds
to be trustworthy (small samples lie — see history below). See memory:
`goal-beat-online-big2-reward`, `eval-needs-online-data`.

### What we have learned (don't relearn the hard way)
- All pure **self-play** models (V1 base, V3 league) and search-on-top (V2, V5) and
  the human line (V4) **lose to humans**, but the gap halved over iterations:
  V1 −9.1 → V5 −3.8/round, win 9% → 18%. V5 (= V4 + uniform PIMC search) is the best
  to date (−3.78, 18%, 50 rounds).
- **Self-play does not teach human-beating play.** Self-play opponents play a
  self-consistent but *alien* style; being good against them ≠ good against humans.
  This is the single most important lesson and the reason for §3.
- **Belief (opponent-hand prediction) is only learnable on the human distribution.**
  On self-play (random deals) it sits at ~chance. Two attempts to convert belief to
  strength failed (belief-conditioned self-play RL; belief-guided PIMC) — see
  memory `ppo-worktree-experiment`. The working use is to **graft the human-trained
  belief onto a determinized full-info search** (the AlphaZero line). See §7.

---

## 2. The one big idea

> **To beat humans, train against opponents that play like humans.**

Everything below follows from this. We build a **pool of opponents** and bias it to
be **as human-like as possible**, because the agent we train against that pool will
specialize in exploiting human-like play — which is the distribution it is deployed
against. As we gather more real human games, we mint better human-imitation
opponents and add them to the pool, so the pool drifts ever closer to "real humans."

---

## 3. The opponent pool

A pool is a set of **frozen** checkpoints used as opponents during league training
(`ppo_trainer.py --pool-ckpts ...`: the learner controls some seats, the rest are
played by random pool members). The machine-readable registry is
[`pool_registry.json`](pool_registry.json) — enumerate/extend it programmatically.

### Humanness — the property we maximize
Each opponent has a **humanness score = BC top-1 match**: the fraction of held-out
real human moves the model would make (greedy) given the same half-masked state.
Higher = more human-like. Measure it the same way `train_bc.py` reports `val top1`
(predict the human's action at each held-out human decision; fraction matched).

| source | humanness | why |
|---|---|---|
| pure imitation (BC on human games) | **high** (~0.74–0.76) | trained to copy humans |
| imitation → light RL | medium | RL drifts away from human style |
| self-play / league | **low** | alien, self-consistent style |

**Pool composition principle:** prefer high-humanness members. Keep a few
self-play members for robustness/variety, but the bulk and the *weighting* should
favor human-like opponents. (Today the trainer samples pool members uniformly;
a worthwhile upgrade is **humanness-weighted sampling** — see §10.)

### Current members
See [`pool_registry.json`](pool_registry.json) for the authoritative list. As of
2026-06-24 the human-trained (preferred) members are `PPO_V4_init` (pure BC, most
human-like), `PPO_V6` (bigger BC + belief), `PPO_V4` (BC→RL); self-play members are
`PPO_V1`, `PPO_V3`. (`PPO_V2` ≡ `PPO_V1` weights; `V5` is a runtime mode, not a file —
see §9 pitfalls.)

### How to add a model to the pool (the iteration step)
1. **Train** the candidate (new BC on the latest human data, or an RL run).
2. **Evaluate** it: offline `evaluate_vs_all` (sanity) **and** measure its
   **humanness** (BC top-1 match on held-out human moves).
3. **Save** the weights to `ppo/checkpoints/saved/<NAME>.pt` (key `"model"`).
4. **Register** it: append an entry to `pool_registry.json` (file, arch, class,
   source, humanness, date_added, trained_on, notes).
5. **Use** it next training round via `--pool-ckpts` (and bias toward it if/when
   humanness-weighted sampling lands).

A new pool member is worth adding when it is **either** more human-like **or**
genuinely different/stronger than existing members (variety helps league training).

---

## 4. The data flywheel (this is what compounds)

```
        play online vs humans (神來也)          ← the only ground truth
                  │  every game auto-recorded + parsed
                  ▼
   ppo/data/online_games.jsonl   (accumulating, dedup, seat-tagged)   ← the asset
                  │  retrain on the LARGER dataset
                  ▼
   better human-imitation opponents (BC)  +  better belief model
                  │  add to pool / swap in
                  ▼
   train the agent against an ever-more-human pool
                  │  deploy
                  ▼
        play online vs humans  ──────────────────►  (loop)
```

Every loop, the dataset grows → imitation opponents get more human-like → the pool
gets more human-like → the trained agent gets better at beating humans → more/better
online games. **The human data is the compounding asset; never throw it away.**

### The data is already accumulated for you — keep it that way
- **Always test online via** [`Big2VisionAgent-claude/play_and_parse.sh`](../../Big2VisionAgent-claude/play_and_parse.sh).
  It plays, tags the version, then runs `record_run` + `parse_online_games`, which
  **append new games (dedup by content `id`)** to `online_games.jsonl`. Data survives
  even after raw artifacts are pruned. (As of 2026-06-24: **578 games / 26,205
  decisions**.)
- **Safety net:** `python -m ppo.parse_online_games` re-scans *all* Big2VisionAgent
  artifacts and picks up anything that was played ad-hoc (without the script). Run it
  if in doubt — it is idempotent.
- **Seat tags:** each game records `seats` = who played each seat (e.g.
  `["V4","human","human","human"]`) so we always know which rows are our agent vs
  real humans. `our_seat` marks our seat.
- Schema, card-id convention, and the **half-masked-obs invariant** are in
  [`data/README.md`](data/README.md). Read it before training on this data.

---

## 5. The belief model (separate from the pool)

`saved/BELIEF.pt` (class `BeliefNet`, `ppo/belief_model.py`) predicts each
opponent's hidden hand `(3,52)` from public info. **It is NOT a pool opponent** — it
is a tool for **search**: importance-sample determinized worlds from it so a
full-info value net evaluates *likely* worlds instead of uniform-random ones.

- Trained on human showdown data → **82.8% held-out P@count** (chance = 33%),
  92–94% late-game. It is strong exactly where it matters (mid/late game) and
  honestly ~chance at the opening (random deals — nothing to predict).
- Intended home: **graft onto the AlphaZero line** (`AlphaBig2-claude`), whose
  `Big2ValueNet` already takes `opp_hands (3,52)` as input. ⚠️ Ordering differs —
  `BeliefNet` is clockwise-relative; the AlphaZero side is ascending-absolute-index.
  Convert seats when grafting. (The AlphaZero net's own belief head is an untrained
  auxiliary, ~chance — do not use it; use `BELIEF.pt`.)
- See memory `ppo-worktree-experiment` for the full belief saga.

**The human data also trains the AlphaZero full-info value net.**
`bc_dataset.build_value_records()` turns each human game into `Big2ValueNet`
(V9 god-view) training states: `encode_static` + `encode_opp_hands` inputs paired
with the game's final per-player score (tanh-normalized, absolute index), with a
game-level train/val split. So the same accumulating dataset (§4) grounds BOTH the
AlphaZero search's **value** AND its **belief** in real human play — exactly the two
pieces that search needs. This is the bridge from this PPO line to the AlphaZero line.

---

## 6. Why not just self-play harder?

We tried (V1 1500-update self-play, V3 league). It plateaus against the *bots* and
still loses to *humans*, because self-play optimizes for beating self-play, a
distribution humans are not drawn from. The pool-of-human-like-opponents approach
re-aims the optimization at the distribution we actually care about. This is the
whole thesis of this file.

---

## 7. Training against the pool — how

League self-play with frozen pool opponents (already implemented in
`ppo/ppo_trainer.py`):

```bash
# warm-start from a human-imitation model, train vs a human-like pool, grow the
# pool with periodic snapshots of the current policy.
./.venv/bin/python -m ppo.ppo_trainer \
  --arch cardaware \
  --init-ckpt ppo/checkpoints/saved/PPO_V4_init.pt \
  --pool-ckpts ppo/checkpoints/saved/PPO_V4_init.pt,ppo/checkpoints/saved/PPO_V6_AS_CARDAWARE_PLACEHOLDER \
  --learner-seats 2 --snapshot-every 50 --updates 400 --tag <newtag>
```

Notes for whoever wires this:
- `--pool-ckpts` are loaded by `_load_frozen`, which reads each checkpoint's own
  `arch`. Mixed-arch pools work **only if** `make_policy`/`_load_frozen` supports
  that arch. Today `cardaware` and `mlp` are supported; **`v6` is not yet wired into
  `policies.py`** — to use `PPO_V6` as a pool opponent you must add a `v6` policy
  adapter (or convert it). Until then, pool with the `cardaware` members.
- Bias toward human-like members (humanness-weighted sampling) is the highest-value
  upgrade to `collect()` — see §10.

---

## 8. Invariants & pitfalls (do not violate)

- **Half-masked obs.** Training observations may use ONLY what the acting player can
  see: own hand + public played cards + opponent **counts**. Never feed opponents'
  hidden card identities into the policy/value obs. Full hands are used only to (a)
  reconstruct the acting player's own hand + legal moves, and (b) as the **belief
  loss oracle** (loss-only, never in obs). This is verified for `bc_dataset.py` and
  `network_cardaware.obs_from_env`. Break this and the model is useless online.
- **`PPO_V2.pt` ≡ `PPO_V1.pt`** (V2's strength was runtime search, not new weights).
  **`V5` has no weight file** — it is `PPO_V4.pt` + uniform PIMC search at inference.
  Don't treat them as distinct networks.
- **Load the right class per checkpoint:** `PPO_V1..V4(+_init)` → `CardAwareActorCritic`
  (`network_cardaware.py`); `PPO_V6` → `CardAwareV6` (`network_v6.py`); `BELIEF` →
  `BeliefNet` (`belief_model.py`). All load via `torch.load(path)["model"]`. The
  registry records the class for each.
- **Small online samples lie.** V5 read −1.12 at 25 rounds, −3.78 at 50. Require
  ~50 rounds before trusting an online verdict.
- **Engine is shared, not copied.** This worktree reuses `AlphaBig2-claude`'s engine
  (`from engine... import`). Run from the worktree root so relative paths resolve.

---

## 9. Quick reference (paths & commands)

| What | Where |
|---|---|
| Accumulated human dataset | `ppo/data/online_games.jsonl` (+ `data/README.md` schema) |
| Version tags per online run | `ppo/data/run_versions.json` |
| Opponent registry (machine-readable) | `ppo/pool_registry.json` |
| Saved models | `ppo/checkpoints/saved/` (`POOL_README.md` = provenance notes) |
| Belief model | `ppo/checkpoints/saved/BELIEF.pt` (class `BeliefNet`) |
| Online test + auto-accumulate | `Big2VisionAgent-claude/play_and_parse.sh <model> <label> [search] [games]` |
| Belief-guided online test | `BELIEF=1 .../play_and_parse.sh PPO_V4.pt V6 1` |
| Online stats by version | `python -m ppo.online_stats` |
| Build BC/belief dataset | `ppo/bc_dataset.py` (`build_records(target=...)`) |
| Train imitation opponent | `python -m ppo.train_bc --target human --epochs 30` |
| Train belief model | `python -m ppo.train_belief --target all --epochs 40` |
| League training | `python -m ppo.ppo_trainer --pool-ckpts ... --snapshot-every 50` |

```bash
# typical iteration
python -m ppo.parse_online_games                 # 1. fold in latest human games (safety net)
python -m ppo.train_bc --target human --epochs 30   # 2. mint a more human-like opponent
#   ... measure humanness + strength, save to saved/<NAME>.pt, append to pool_registry.json
python -m ppo.ppo_trainer --init-ckpt ... --pool-ckpts ... --tag <new>   # 3. train vs the pool
#   ... deploy: BELIEF=1 play_and_parse.sh saved/<new>.pt <label> 1       # 4. test online -> more data
```

---

## 10. Roadmap / open questions (for the next agent)

1. **Humanness-weighted pool sampling.** `collect()` in `ppo_trainer.py` samples pool
   members uniformly. Weight by humanness so training emphasizes human-like
   opponents. Highest-value, lowest-risk upgrade.
2. **Wire `v6` arch into `policies.py`** so `PPO_V6` can serve as a pool opponent
   (currently only `cardaware`/`mlp` load as frozen opponents).
3. **A `measure_humanness.py`** helper (greedy match vs held-out human moves) so
   step 2 of "add to pool" is one command, and humanness is auto-filled in the registry.
4. **Graft `BELIEF.pt` onto the AlphaZero line** (`AlphaBig2-claude`): importance-
   sample determinizations for `Big2ValueNet` from the belief (convert seat ordering).
   This is the most promising path to actually using the belief.
5. **Periodic opponent refresh.** As the dataset grows (e.g. every +200 human games),
   retrain the imitation opponents + belief and add the new versions to the pool.
