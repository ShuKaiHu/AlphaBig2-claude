"""Fitted Value Iteration: extend the exact best-play value past the solvable
endgame frontier (k<=10) into mid-game, by bootstrapping the net on itself.

Anchor   : exact-solved k<=10 positions (fixed ground-truth targets).
Bootstrap: deeper positions whose target is a 1-ply max^n backup using the
           CURRENT net at the children (terminal children use the true reward).
           Iterating propagates the exact terminal signal outward through V.
Validate : a held-out set of EXACT-solved k=12 positions (ground truth the net
           was NEVER trained on at that depth) — if FVI works, the net predicts
           them well below the predict-mean baseline.

  python -m ppo.fvi
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn

from engine.env import Big2Env, PASS_IDX
from engine.features import encode_static, encode_opp_hands
from engine.model import Big2ValueNet
from ppo.endgame_value import _random_endgame, solve, REWARD_SCALE

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def expand(env):
    """(acting, static, opp, children) — children: list of dicts
    {'term': reward4}  OR  {'cs': child_static, 'co': child_opp}."""
    g = env.game; acting = env.current_player
    static = encode_static(g, acting).astype(np.float32)
    opp = encode_opp_hands(g, acting).astype(np.float32)
    valid = np.flatnonzero(env.get_valid_actions())
    if len(valid) == 0:
        valid = np.array([PASS_IDX])
    children = []
    for a in valid:
        e2 = env.clone(); rw, done = e2.step(int(a))
        if done:
            children.append({"term": np.asarray(rw, np.float32)})
        else:
            ca = e2.current_player
            children.append({"cs": encode_static(e2.game, ca).astype(np.float32),
                             "co": encode_opp_hands(e2.game, ca).astype(np.float32)})
    return acting, static, opp, children


def build_pool(n, k_range, seed):
    rng = np.random.default_rng(seed)
    pool = []
    while len(pool) < n:
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        env = _random_endgame(k, rng)
        if env is None:
            continue
        acting, static, opp, children = expand(env)
        if not children:
            continue
        pool.append({"acting": acting, "static": static, "opp": opp, "children": children})
    return pool


def build_exact_val(n, k, seed):
    """Held-out EXACT-solved positions (ground truth)."""
    rng = np.random.default_rng(seed)
    S, O, Y = [], [], []
    while len(S) < n:
        env = _random_endgame(k, rng)
        if env is None:
            continue
        acting = env.current_player
        val = solve(env, {})
        S.append(encode_static(env.game, acting).astype(np.float32))
        O.append(encode_opp_hands(env.game, acting).astype(np.float32))
        Y.append(np.tanh(val / REWARD_SCALE).astype(np.float32))
    return np.stack(S), np.stack(O), np.stack(Y)


@torch.no_grad()
def backup_targets(model, pool):
    """1-ply max^n backup target (4-dim tanh) for each pool position, using the
    current model at non-terminal children."""
    # gather all net-children features into one batch
    feats_s, feats_o, slot = [], [], []
    for pi, p in enumerate(pool):
        for ci, ch in enumerate(p["children"]):
            if "cs" in ch:
                slot.append((pi, ci)); feats_s.append(ch["cs"]); feats_o.append(ch["co"])
    cv = {}
    if feats_s:
        S = torch.from_numpy(np.stack(feats_s)); O = torch.from_numpy(np.stack(feats_o))
        out = model(S, O).numpy()  # (M,4) tanh
        for (pi, ci), v in zip(slot, out):
            cv[(pi, ci)] = v
    targets = np.zeros((len(pool), 4), dtype=np.float32)
    for pi, p in enumerate(pool):
        acting = p["acting"]
        best = None
        for ci, ch in enumerate(p["children"]):
            v = np.tanh(ch["term"] / REWARD_SCALE) if "term" in ch else cv[(pi, ci)]
            if best is None or v[acting - 1] > best[acting - 1]:
                best = v
        targets[pi] = best
    return targets


def main():
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)

    print("building exact k=12 validation set (ground truth)...")
    VS, VO, VY = build_exact_val(80, k=12, seed=999)
    VS, VO, VY = map(torch.from_numpy, (VS, VO, VY))
    base = ((VY - VY.mean(0)) ** 2).mean().item()
    print(f"  val baseline MSE (predict-mean) = {base:.4f}  ({time.time()-t0:.0f}s)")

    # anchor: exact k<=10 (from the saved npz)
    d = np.load(os.path.join(os.path.dirname(__file__), "data", "endgame_solved_k10.npz"))
    AS = torch.from_numpy(d["static"]); AO = torch.from_numpy(d["opp"]); AY = torch.from_numpy(d["target"])
    print(f"anchor (exact k<=10): {len(AS)}")

    print("building bootstrap pool k=11..16...")
    pool = build_pool(4000, (11, 16), seed=1)
    PS = torch.from_numpy(np.stack([p["static"] for p in pool]))
    PO = torch.from_numpy(np.stack([p["opp"] for p in pool]))
    print(f"  pool: {len(pool)}  ({time.time()-t0:.0f}s)")

    model = Big2ValueNet()
    # warm start from the k<=10 exact-trained net if present
    seed_path = os.path.join(CKPT_DIR, "value_endgame_best.pt")
    if os.path.exists(seed_path):
        model.load_state_dict(torch.load(seed_path, map_location="cpu")["value_state"])
        print("warm-started from value_endgame_best.pt")
    opt = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    lossf = nn.MSELoss()

    def val_mse():
        model.eval()
        with torch.no_grad():
            m = lossf(model(VS, VO), VY).item()
        model.train(); return m

    print(f"FVI start | val MSE {val_mse():.4f} vs baseline {base:.4f}")
    for it in range(1, 13):
        PT = torch.from_numpy(backup_targets(model, pool))   # recompute targets w/ current net
        # one training pass over anchor + bootstrapped pool
        Xs = torch.cat([AS, PS]); Xo = torch.cat([AO, PO]); Y = torch.cat([AY, PT])
        idx = np.random.permutation(len(Xs))
        model.train()
        for ep in range(8):
            np.random.shuffle(idx)
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                loss = lossf(model(Xs[b], Xo[b]), Y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        vm = val_mse()
        print(f"  FVI iter {it:2d} | exact-k12 val MSE {vm:.4f}"
              f"  {'✓<baseline' if vm < base else '✗'}  ({time.time()-t0:.0f}s)")

    torch.save({"value_state": {k: v.clone() for k, v in model.state_dict().items()},
                "scale": REWARD_SCALE, "tag": "value_fvi"},
               os.path.join(CKPT_DIR, "value_fvi_best.pt"))
    print(f"saved -> value_fvi_best.pt | final val MSE {val_mse():.4f} vs baseline {base:.4f}")


if __name__ == "__main__":
    main()
