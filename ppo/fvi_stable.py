"""Stabilized Fitted Value Iteration (fixes the divergence in ppo/fvi.py).

Naive FVI diverged (deadly triad: bootstrap + function approx + self-reference):
targets recomputed from the live net every step → positive feedback → drift past
baseline. Three standard fixes here:
  1. TARGET NETWORK — backup targets come from a FROZEN copy, refreshed slowly,
     so the net isn't chasing a target that moves every step.
  2. EARLY STOP on the exact-solved k=12 validation (real ground truth) — keep
     the best net, never the diverged one.
  3. SMALL STEPS — few epochs per refresh.

Slow data (exact val + bootstrap pool) is cached to disk so the FVI loop is fast
to re-tune.

  python -m ppo.fvi_stable
"""
import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn

from engine.model import Big2ValueNet
from engine.features import encode_static, encode_opp_hands
from ppo.endgame_value import _random_endgame, solve, REWARD_SCALE
from ppo.fvi import expand

CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
CACHE = os.path.join(os.path.dirname(__file__), "data", "fvi_cache.pkl")


def _exact_val(n, k, seed):
    rng = np.random.default_rng(seed); S, O, Y = [], [], []
    while len(S) < n:
        env = _random_endgame(k, rng)
        if env is None:
            continue
        a = env.current_player; val = solve(env, {})
        S.append(encode_static(env.game, a).astype(np.float32))
        O.append(encode_opp_hands(env.game, a).astype(np.float32))
        Y.append(np.tanh(val / REWARD_SCALE).astype(np.float32))
    return np.stack(S), np.stack(O), np.stack(Y)


def build_cache(n_pool=5000, k_range=(11, 18), n_val=80, seed=999):
    t0 = time.time()
    print("exact k=12 val..."); vs, vo, vy = _exact_val(n_val, 12, seed)
    print(f"  done ({time.time()-t0:.0f}s)")
    print("bootstrap pool k=11..18...")
    rng = np.random.default_rng(1)
    pos_s, pos_o, pos_a = [], [], []
    child_s, child_o = [], []            # global flat list of net-children
    per_term = []                        # per position: list of 4-dim tanh reward
    per_netidx = []                      # per position: list of indices into child arrays
    while len(pos_s) < n_pool:
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        env = _random_endgame(k, rng)
        if env is None:
            continue
        a, st, op, children = expand(env)
        if not children:
            continue
        terms, nidx = [], []
        for ch in children:
            if "term" in ch:
                terms.append(np.tanh(ch["term"] / REWARD_SCALE).astype(np.float32))
            else:
                nidx.append(len(child_s)); child_s.append(ch["cs"]); child_o.append(ch["co"])
        pos_s.append(st); pos_o.append(op); pos_a.append(a)
        per_term.append(terms); per_netidx.append(np.array(nidx, dtype=np.int64))
        if len(pos_s) % 500 == 0:
            print(f"  pool {len(pos_s)}/{n_pool} (childs {len(child_s)}, {time.time()-t0:.0f}s)")
    cache = {
        "val": (vs, vo, vy),
        "pos_s": np.stack(pos_s), "pos_o": np.stack(pos_o),
        "pos_a": np.array(pos_a, dtype=np.int64),
        "child_s": np.stack(child_s), "child_o": np.stack(child_o),
        "per_term": per_term, "per_netidx": per_netidx,
    }
    with open(CACHE, "wb") as f:
        pickle.dump(cache, f)
    print(f"cached -> {CACHE}  ({time.time()-t0:.0f}s)")
    return cache


def main():
    if os.path.exists(CACHE):
        cache = pickle.load(open(CACHE, "rb"))
        print("loaded cache")
    else:
        cache = build_cache()

    vs, vo, vy = (torch.from_numpy(x) for x in cache["val"])
    base = ((vy - vy.mean(0)) ** 2).mean().item()
    PS = torch.from_numpy(cache["pos_s"]); PO = torch.from_numpy(cache["pos_o"])
    pos_a = cache["pos_a"]; per_term = cache["per_term"]; per_netidx = cache["per_netidx"]
    CS = torch.from_numpy(cache["child_s"]); CO = torch.from_numpy(cache["child_o"])
    d = np.load(os.path.join(os.path.dirname(__file__), "data", "endgame_solved_k10.npz"))
    AS = torch.from_numpy(d["static"]); AO = torch.from_numpy(d["opp"]); AY = torch.from_numpy(d["target"])
    print(f"anchor {len(AS)} | pool {len(PS)} (childs {len(CS)}) | val baseline {base:.4f}")

    online = Big2ValueNet()
    seed_path = os.path.join(CKPT_DIR, "value_endgame_best.pt")
    if os.path.exists(seed_path):
        online.load_state_dict(torch.load(seed_path, map_location="cpu")["value_state"])
        print("warm-started from value_endgame_best.pt")
    target = Big2ValueNet(); target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=3e-4, weight_decay=1e-4)
    lossf = nn.MSELoss()

    def val_mse(net):
        net.eval()
        with torch.no_grad():
            m = lossf(net(vs, vo), vy).item()
        net.train(); return m

    def backup_with(net):
        net.eval()
        with torch.no_grad():
            Vc = net(CS, CO).numpy()  # (M,4) tanh, computed by TARGET net
        net.train()
        T = np.zeros((len(per_term), 4), dtype=np.float32)
        for i in range(len(per_term)):
            acting = pos_a[i]
            cands = list(per_term[i]) + [Vc[j] for j in per_netidx[i]]
            best = cands[int(np.argmax([c[acting - 1] for c in cands]))]
            T[i] = best
        return torch.from_numpy(T)

    REFRESH = 3          # refresh target net every 3 iters (slow target)
    EPOCHS = 2           # small steps per iter
    best_v, best_state, since = val_mse(online), None, 0
    best_state = {k: v.clone() for k, v in online.state_dict().items()}
    print(f"start | val {best_v:.4f} vs baseline {base:.4f}")

    for it in range(1, 31):
        PT = backup_with(target)                       # targets from FROZEN target net
        Xs = torch.cat([AS, PS]); Xo = torch.cat([AO, PO]); Y = torch.cat([AY, PT])
        idx = np.arange(len(Xs)); online.train()
        for _ in range(EPOCHS):
            np.random.shuffle(idx)
            for i in range(0, len(idx), 256):
                b = idx[i:i + 256]
                loss = lossf(online(Xs[b], Xo[b]), Y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        if it % REFRESH == 0:
            target.load_state_dict(online.state_dict())  # slow target refresh
        vm = val_mse(online)
        imp = vm < best_v - 1e-5
        if imp:
            best_v, since = vm, 0
            best_state = {k: v.clone() for k, v in online.state_dict().items()}
        else:
            since += 1
        tgt = " [refresh]" if it % REFRESH == 0 else ""
        print(f"iter {it:2d} | val {vm:.4f} {'✓' if vm < base else '✗'}{tgt}"
              f"{'  *best' if imp else ''}")
        if since >= 8:
            print(f"early stop @ {it}"); break

    online.load_state_dict(best_state)
    torch.save({"value_state": best_state, "scale": REWARD_SCALE, "tag": "value_fvi_stable"},
               os.path.join(CKPT_DIR, "value_fvi_stable_best.pt"))
    print(f"saved -> value_fvi_stable_best.pt | BEST val {best_v:.4f} vs baseline {base:.4f}"
          f"  → {'✓ stable & beats baseline' if best_v < base else '✗'}")


if __name__ == "__main__":
    main()
