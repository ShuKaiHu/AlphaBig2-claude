# AlphaBig2-Value — AlphaGo-style value model for Big 2

> **Audience:** the author + future AI coding agents.
> A value model that, from **public information only**, predicts the **final score**
> (how much you'll win/lose) of the current position — trained the AlphaGo way:
> self-play a policy for many games, then learn from (position → game outcome) pairs.

---

## What it is

`V(public_state) → expected final score` for the player to move.

- **Input:** the acting player's PUBLIC view — own hand, all played cards,
  per-opponent played cards, opponent counts, current trick, pass count. (The
  cardaware obs, `ppo.network_cardaware.obs_from_env`.) No hidden cards.
- **Output:** scalar in `[-1, 1]` = `tanh(final_score / 13)`. Multiply by 13 for the
  real 神來也-style score (`ValueNet.value_from_obs` does this).
- **Why public-info (imperfect):** this is the AlphaGo value-net analogue for an
  imperfect-information game. It learns `E[final score | what you can see, both
  sides keep playing the policy]`, averaging over the hidden hands. Unlike the
  full-info `Big2ValueNet` in `AlphaBig2-claude` (which needs the opponents' actual
  hands and is meant for *determinized* search), this one evaluates a position
  directly with no determinization.

## The AlphaGo recipe (3 steps)

1. **Self-play** (`selfplay.py`): all four seats are played by a **human-data
   policy** (default `PPO_V4.pt`), **sampling** from its softmax (not greedy — greedy
   replays one identical game) so we get a diverse, human-like game corpus. Each game
   is saved (hands + plays + scores) to `data/selfplay_games.jsonl`, **appending** so
   runs accumulate.
2. **Dataset** (`value_dataset.py`): for each game pick **ONE random position** and
   label it with that player's **final score**. *One sample per game* is the crucial
   AlphaGo trick — positions within a game are highly correlated and share the label,
   so using many per game overfits. (`samples_per_game` defaults to 1; raise only if
   you understand the trade-off.)
3. **Train** (`train_value.py`): MSE on `tanh(score/13)`. Watch **val MSE vs baseline
   MSE** (= variance of targets; beating it = learning), **Pearson r**, and **sign
   accuracy** (win/lose direction). Note: public-info value has irreducible variance
   (the same public state can end many ways), so don't expect tiny MSE — expect MSE
   below baseline with positive r.

## Quick start

```bash
PY=/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/.venv/bin/python
cd /Users/shukaihu/Code_Project_Local/AlphaBig2-Value
$PY selfplay.py --games 40000 --policy PPO_V4.pt   # step 1 (accumulates)
$PY train_value.py --epochs 40                     # steps 2+3 (1 sample/game)
```

| File | Role |
|---|---|
| `bootstrap.py` | path setup — reuses AB2PPO engine/policy (adds to `sys.path`, chdir for `actionIndices.pkl`); exports absolute `DATA_DIR`/`CKPT_DIR` |
| `selfplay.py` | generate + save self-play games (step 1) |
| `value_dataset.py` | AlphaGo 1-position-per-game sampler (step 2) |
| `value_model.py` | `ValueNet` (public obs → tanh score) + `value_from_obs` |
| `train_value.py` | train + eval (step 3) |
| `data/selfplay_games.jsonl` | accumulating game corpus (same schema as the human `online_games.jsonl`) |
| `checkpoints/value_best.pt` | best value net (by val MSE) |

## Design notes / gotchas

- **Engine is reused, not copied.** `bootstrap.py` puts `AlphaBig2-ppo` on the path
  and chdirs there (so `enumerateOptions` finds `actionIndices.pkl`); our outputs use
  absolute paths. Run scripts with the AB2PPO venv python.
- **Self-play policy choice defines the value.** Default `PPO_V4.pt` (human-data
  BC→RL, our deployed policy). For a more human-like value use `--policy PPO_V4_init.pt`
  (pure imitation); for the bigger net use `--policy PPO_V6.pt`. The value net learns
  the value of *that* policy's play.
- **Card-id / schema** match the rest of the project (`id=(rank-1)*4+suit`; game
  schema = `online_games.jsonl`), so self-play games are interchangeable with the
  human dataset for tooling.
- **Relationship to the other lines:** the `AlphaBig2-ppo` line owns the policy + the
  opponent pool + the human-trained **belief** model; `AlphaBig2-claude` owns the
  AlphaZero search + a **full-info** value net. This workspace adds a **public-info**
  value net trained AlphaGo-style on policy self-play. See
  `AlphaBig2-ppo/ppo/TRAINING_PLAN.md` for the overall strategy.
