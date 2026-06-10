"""
Main training loop for AlphaBig2.

Usage:
    python -m engine.trainer
    python -m engine.trainer --iterations 500 --self-play 100 --sims 50
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F

from engine.model import Big2Net, Big2ValueNet
from engine.self_play import run_episode, make_pool_opponent
from engine.replay_buffer import ReplayBuffer
from engine.evaluator import evaluate_vs_heuristic

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train(
    n_iterations: int = 1000,
    self_play_episodes: int = 200,
    train_steps: int = 500,
    batch_size: int = 256,
    lr: float = 2e-4,
    n_simulations: int = 50,
    replay_capacity: int = 50_000,
    pool_update_freq: int = 50,
    eval_freq: int = 5,
    eval_games: int = 200,
    checkpoint_dir: str = CHECKPOINT_DIR,
    resume: bool = True,
    bc_warmup_iters: int = 50,
    bc_mix_ratio: float = 0.0,  # fraction of episodes always using BC (even after warmup)
    entropy_coef: float = 0.0,  # entropy BONUS weight. 0 = off (AlphaZero-style:
                                # rely on Dirichlet noise, not an entropy bonus).
                                # A positive bonus drove entropy runaway here.
    policy_target_temp: float = 0.7,  # <1 sharpens MCTS visit targets to counter
                                      # the high variance of few-sim visit counts.
    value_coef: float = 1.0,   # weight on the value MSE. The raw v_loss (~0.02)
                               # is dwarfed by p_loss (~2.7), so value gets <1%
                               # of the gradient and stays weak/under-confident.
                               # Up-weighting (e.g. 3-5) gives the value head more
                               # learning signal.
    belief_coef: float = 0.1,  # weight on the opponent-hand belief auxiliary loss.
                               # Raise it to make the belief head genuinely
                               # predictive (needed for belief-guided
                               # determinization sampling at inference).
    league: bool = False,      # train some episodes vs frozen PAST versions
                               # (anti-collapse diversity), not just mirror-self.
    league_ratio: float = 0.5, # fraction of (post-warmup) episodes using pool opps.
    full_info_value: bool = False,  # V9: train a god-view Big2ValueNet (input =
                                    # static + true opponent hands) and use it for
                                    # MCTS leaf evaluation in self-play. The policy
                                    # net (Big2Net) stays imperfect-info and its own
                                    # value head keeps training as before (ablation/
                                    # fallback). Training input reuses the stored
                                    # belief_target (same 3×52 layout).
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, "latest.pt")
    best_path = os.path.join(checkpoint_dir, "best.pt")

    model = Big2Net()
    value_net = Big2ValueNet() if full_info_value else None
    params = list(model.parameters())
    if value_net is not None:
        params += list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_iterations, eta_min=lr * 0.1
    )
    buffer = ReplayBuffer(replay_capacity)
    opponent_pool: list = []
    best_avg_score = -float("inf")
    start_iter = 1

    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu")
        # Filter out parameters whose shape no longer matches (e.g. value_head
        # changed from 1-dim to 4-dim during the multi-player rebuild). strict=False
        # only tolerates missing/extra KEYS — a shape mismatch on a shared key
        # still raises, so we drop those keys and let them re-initialize.
        old_state = ckpt["model_state"]
        cur_state = model.state_dict()
        compatible = {
            k: v for k, v in old_state.items()
            if k in cur_state and cur_state[k].shape == v.shape
        }
        dropped = [k for k in old_state if k not in compatible]
        missing, unexpected = model.load_state_dict(compatible, strict=False)
        if dropped:
            print(f"Re-initialized (shape changed): {dropped}")
        if missing:
            print(f"New weights (randomly init): {missing}")
        if unexpected:
            print(f"Ignored old keys: {unexpected}")
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except ValueError:
            print("Optimizer state incompatible (architecture changed), starting fresh optimizer.")
        if value_net is not None and "value_state" in ckpt:
            value_net.load_state_dict(ckpt["value_state"])
        best_avg_score = ckpt.get("best_avg_score", -float("inf"))
        start_iter = ckpt.get("iteration", 0) + 1
        print(f"Resumed from iteration {start_iter - 1}, best_avg_score={best_avg_score:.3f}")
        # Restore scheduler to correct position so LR doesn't restart from peak
        for _ in range(start_iter - 1):
            scheduler.step()

    for iteration in range(start_iter, n_iterations + 1):
        t0 = time.time()
        temperature = max(0.6, 1.0 - iteration / 300.0)

        # ── Self-play ─────────────────────────────────────────────────────────
        model.eval()
        if value_net is not None:
            value_net.eval()
        ep_rewards = []
        bc_active = (iteration <= bc_warmup_iters)
        n_bc_mix = int(self_play_episodes * bc_mix_ratio) if not bc_active else 0
        for ep_idx in range(self_play_episodes):
            use_bc = bc_active or (ep_idx < n_bc_mix)
            # League: for some episodes, seats 2-4 are frozen PAST versions
            # (diverse opponents) so the learner (seat 1) is forced to generalize
            # instead of collapsing to a narrow mirror-self equilibrium. The rest
            # stay pure 4-player self-play (all seats contribute data).
            opp = None
            if (league and opponent_pool and not use_bc
                    and np.random.rand() < league_ratio):
                opp = {s: make_pool_opponent(np.random.choice(opponent_pool))
                       for s in (2, 3, 4)}
            data, reward = run_episode(
                model,
                n_simulations=n_simulations,
                temperature=temperature,
                opponent=opp,
                bc_mode=use_bc,
                value_model=value_net,
            )
            buffer.add_episode(data)
            ep_rewards.append(reward)

        avg_reward = float(np.mean(ep_rewards))

        # ── Train ─────────────────────────────────────────────────────────────
        if len(buffer) < batch_size:
            print(
                f"[iter {iteration:4d}] buffer too small ({len(buffer)}), skipping train"
            )
            continue

        model.train()
        if value_net is not None:
            value_net.train()
        p_losses, v_losses, b_losses, entropies, vf_losses = [], [], [], [], []
        for _ in range(train_steps):
            statics, histories, valids, pol_tgts, val_tgts, bel_tgts = buffer.sample(batch_size)
            sf = torch.FloatTensor(statics)
            hs = torch.FloatTensor(histories)
            vm = torch.FloatTensor(valids)
            pt = torch.FloatTensor(pol_tgts)
            vt = torch.FloatTensor(np.asarray(val_tgts))   # (B, 4)
            bt = torch.FloatTensor(bel_tgts)

            logits, value, belief_logits = model(sf, hs, vm)

            # Policy: cross-entropy vs MCTS visit distribution.
            # 1) Sharpen the target (temp<1) to counter the high variance of
            #    few-sim visit counts spread over a large action space —
            #    without this the targets are too diffuse and entropy drifts up.
            # 2) Light label smoothing (eps=0.05) for stability.
            log_probs = F.log_softmax(logits, dim=-1)
            if policy_target_temp != 1.0:
                pt_sharp = pt.clamp(min=0) ** (1.0 / policy_target_temp)
                pt = pt_sharp / pt_sharp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            valid_count = vm.sum(dim=-1, keepdim=True).clamp(min=1.0)
            pt_smooth = pt * 0.95 + 0.05 * vm / valid_count
            p_loss = -(pt_smooth * log_probs).sum(dim=-1).mean()

            # Entropy bonus: prevent policy collapse
            # probs over LEGAL actions only (vm mask); encourages exploration
            probs = torch.exp(log_probs) * vm
            probs_sum = probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            probs_norm = probs / probs_sum
            entropy = -(probs_norm * (probs_norm + 1e-8).log()).sum(dim=-1).mean()

            # Value: MSE vs 4-dim Monte-Carlo terminal outcome (per player)
            v_loss = F.mse_loss(value, vt)

            # Belief: BCE vs oracle opponent hands (auxiliary, weight=0.1)
            b_loss = F.binary_cross_entropy_with_logits(belief_logits, bt)

            # entropy_coef defaults to 0 (no bonus): a positive bonus actively
            # inflated entropy here and caused runaway. Exploration comes from
            # MCTS root Dirichlet noise instead.
            loss = p_loss + value_coef * v_loss + belief_coef * b_loss - entropy_coef * entropy

            # V9 full-info value net: same TD(λ) targets, but the input includes
            # the true opponent hands (bel_tgts has exactly that layout).
            if value_net is not None:
                vf = value_net(sf, bt)
                vf_loss = F.mse_loss(vf, vt)
                loss = loss + value_coef * vf_loss
                vf_losses.append(vf_loss.item())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            p_losses.append(p_loss.item())
            v_losses.append(v_loss.item())
            b_losses.append(b_loss.item())
            entropies.append(entropy.item())

        scheduler.step()

        # ── Evaluate ──────────────────────────────────────────────────────────
        mode_label = 'BC' if bc_active else (f'MCTS+{int(bc_mix_ratio*100)}%BC' if bc_mix_ratio > 0 else 'MCTS')
        log_parts = [
            f"[iter {iteration:4d}]",
            mode_label,
            f"reward={avg_reward:+.2f}",
            f"p_loss={np.mean(p_losses):.4f}",
            f"v_loss={np.mean(v_losses):.4f}",
            f"b_loss={np.mean(b_losses):.4f}",
            f"ent={np.mean(entropies):.3f}",
            f"T={temperature:.2f}",
            f"buf={len(buffer)}",
        ]
        if vf_losses:
            # full-info value loss — should run BELOW v_loss (more predictable input)
            log_parts.insert(5, f"vf_loss={np.mean(vf_losses):.4f}")

        if iteration % eval_freq == 0:
            model.eval()
            win_rate, avg_score = evaluate_vs_heuristic(model, n_games=eval_games)
            log_parts += [
                f"| avg_score={avg_score:+.2f}",
                f"win_rate={win_rate:.3f}",
            ]
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                if value_net is not None:
                    torch.save({"model_state": model.state_dict(),
                                "value_state": value_net.state_dict()}, best_path)
                else:
                    torch.save(model.state_dict(), best_path)
                log_parts.append("✓ best")

        elapsed = time.time() - t0
        log_parts.append(f"[{elapsed:.0f}s]")
        print(" ".join(log_parts))

        # ── Opponent pool (league) ───────────────────────────────────────────────
        # Snapshot the current model into the frozen-opponent pool every
        # pool_update_freq iters. Keeps the last 10 versions for diversity.
        if league and iteration % pool_update_freq == 0 and not bc_active:
            snapshot = Big2Net()
            snapshot.load_state_dict(copy.deepcopy(model.state_dict()))
            snapshot.eval()
            opponent_pool.append(snapshot)
            if len(opponent_pool) > 10:
                opponent_pool.pop(0)
            print(f"  Pool updated: {len(opponent_pool)} frozen opponents")

        # ── Save checkpoint ───────────────────────────────────────────────────
        ckpt_out = {
            "iteration": iteration,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_avg_score": best_avg_score,
        }
        if value_net is not None:
            ckpt_out["value_state"] = value_net.state_dict()
        torch.save(ckpt_out, latest_path)

    print(f"\nTraining complete. Best avg_score: {best_avg_score:.3f}")
    print(f"Best model: {best_path}")
    return model


def _parse_args():
    p = argparse.ArgumentParser(description="Train AlphaBig2")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--self-play", type=int, default=200, dest="self_play_episodes")
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--sims", type=int, default=50, dest="n_simulations")
    p.add_argument("--eval-freq", type=int, default=5)
    p.add_argument("--eval-games", type=int, default=200)
    p.add_argument("--replay-capacity", type=int, default=50_000, dest="replay_capacity")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--bc-warmup", type=int, default=50, dest="bc_warmup_iters",
                   help="Number of BC warmup iterations before switching to MCTS self-play")
    p.add_argument("--bc-mix", type=float, default=0.0, dest="bc_mix_ratio",
                   help="Fraction of episodes always using BC after warmup (0.0-1.0)")
    p.add_argument("--entropy-coef", type=float, default=0.0, dest="entropy_coef",
                   help="Entropy BONUS weight (0=off; positive values can cause runaway)")
    p.add_argument("--policy-temp", type=float, default=0.7, dest="policy_target_temp",
                   help="Temperature (<1 sharpens) applied to MCTS visit policy targets")
    p.add_argument("--value-coef", type=float, default=1.0, dest="value_coef",
                   help="Weight on value MSE (raise, e.g. 3-5, to train the value head harder)")
    p.add_argument("--belief-coef", type=float, default=0.1, dest="belief_coef",
                   help="Weight on belief auxiliary loss (raise to make belief predictive)")
    p.add_argument("--league", action="store_true", dest="league",
                   help="Train some episodes vs frozen past versions (anti-collapse)")
    p.add_argument("--league-ratio", type=float, default=0.5, dest="league_ratio",
                   help="Fraction of post-warmup episodes using pool opponents")
    p.add_argument("--full-info-value", action="store_true", dest="full_info_value",
                   help="V9: train a god-view value net (sees all four hands) and use "
                        "it for MCTS leaf evaluation in self-play")
    p.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR, dest="checkpoint_dir",
                   help="Where to write latest.pt/best.pt (use a separate dir to protect deployment)")
    p.add_argument("--torch-threads", type=int, default=3,
                   help="PyTorch intra-op threads (3 = ~30%% of 10-core CPU)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    # 限制 PyTorch 平行度（雙保險：cpulimit 負責系統層，這裡負責 PyTorch 層）
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(args.torch_threads)
    print(f"PyTorch threads: {args.torch_threads} (intra + interop)")
    train(
        n_iterations=args.iterations,
        self_play_episodes=args.self_play_episodes,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        n_simulations=args.n_simulations,
        replay_capacity=args.replay_capacity,
        eval_freq=args.eval_freq,
        eval_games=args.eval_games,
        resume=args.resume,
        bc_warmup_iters=args.bc_warmup_iters,
        bc_mix_ratio=args.bc_mix_ratio,
        entropy_coef=args.entropy_coef,
        policy_target_temp=args.policy_target_temp,
        value_coef=args.value_coef,
        belief_coef=args.belief_coef,
        league=args.league,
        league_ratio=args.league_ratio,
        checkpoint_dir=args.checkpoint_dir,
        full_info_value=args.full_info_value,
    )
