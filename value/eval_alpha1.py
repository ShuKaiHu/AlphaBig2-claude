"""Local internal test: Alpha1 (ISMCTS) vs V4 (greedy policy).

Alpha1 = the new search — PPO_V4 policy + BELIEF.pt + VALUE.pt combined in a true
ISMCTS (single information-set tree keyed by OUR action-path; each simulation
re-determinizes a world from belief; opponents play forward via PPO_V4; our nodes
PUCT with the policy prior; leaf = VALUE.pt from our perspective). This is an
engine-side reimplementation of cardaware_wrapper's ISMCTS (any seat, no vision /
MockGame), used to (a) validate the build and (b) measure strength vs plain V4.

V4 = PPO_V4 greedy (policy only, no search). Since Alpha1 = V4 + search, Alpha1-vs-V4
isolates "does the belief+value ISMCTS make V4 stronger?".

Setup: 1 Alpha1 seat vs 3 V4 seats, Alpha1's seat rotated across games (Big2 has
seat/turn-order effects). IMPORTANT: Alpha1 only ever uses its own PUBLIC view +
belief determinization — it never reads opponents' true hands (no cheating); the
engine knows the true hands only to resolve the game.

    python eval_alpha1.py --games 40 --sims 80
"""
import bootstrap  # noqa: F401  (engine + ppo on path; chdir for actionIndices.pkl)

import argparse
import os
import time

import numpy as np
import torch

from engine.env import Big2Env, PASS_IDX
from ppo.network_cardaware import (CardAwareActorCritic, obs_from_env,
                                   legal_action_data, _to_batch)
from ppo.action_features import ACTION_FEATURES
from ppo.belief_model import BeliefNet, belief_from_obs
from value_model import ValueNet, SCALE
from dominance_model import DominanceNet
from hand_features import min_plays_to_empty

DEV = "cpu"
_LEAF = "value"        # "value" = VALUE.pt leaf; "rollout" = greedy-to-terminal leaf
_VALUE_DEPTH = 0       # value leaf: simulate this many greedy rounds forward, THEN VALUE.pt
_USE_BELIEF = True     # determinize worlds from BELIEF.pt (False = uniform random worlds)
_BELIEF_OVERRIDE = None  # optional fn(env)->(3,52); external harness swaps belief source
_OVR = [0, 0]          # [search-action != greedy-action count, total decisions] — override rate

# De-anchoring (default OFF — cardaware_wrapper.py's deployed ISMCTS is untouched unless these
# are explicitly enabled). Diagnosed this session: PUCT's cpuct*prior*sqrt(N_avail)/(1+N) term
# starves low-prior root actions of simulation budget regardless of true value — a direct
# rollout-vs-policy comparison (no PUCT involved) found ~25% of real decisions have a CLEAR
# (non-noise) policy miss, far more than the ~7.8% ISMCTS actually overrides. These two flags
# fix that at the ROOT only (root exploration is the well-precedented place for both, AlphaZero
# and KataGo-style): every legal root action keeps some exploration mass regardless of prior.
_ROOT_NOISE = False        # mix Dirichlet noise into the root prior (AlphaZero-style)
_ROOT_NOISE_EPS = 0.25     # weight of the noise vs the policy prior
_ROOT_NOISE_ALPHA = 0.3    # Dirichlet concentration
_MIN_VISITS = 0            # force every root action to >=N visits before PUCT free-selection (0=off)
torch.set_num_threads(1)
SAVED = "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo/ppo/checkpoints/saved"
VALUE_CKPT = "/Users/shukaihu/Code_Project_Local/AlphaBig2-Value/checkpoints/VALUE.pt"

_policy = CardAwareActorCritic().to(DEV)
_policy.load_state_dict(torch.load(f"{SAVED}/PPO_V4.pt", map_location=DEV)["model"]); _policy.eval()
_belief = BeliefNet().to(DEV)
_belief.load_state_dict(torch.load(f"{SAVED}/BELIEF.pt", map_location=DEV)["model"]); _belief.eval()
_value = ValueNet().to(DEV)
_value.load_state_dict(torch.load(VALUE_CKPT, map_location=DEV)["model"]); _value.eval()
_value_aug = None      # ValueNet(extra_dim=4): VALUE + dominance probs + min-plays (4-dim)
_value_mp = None       # ValueNet(extra_dim=1): VALUE + min-plays only — the DEPLOY value
_dom = None            # DominanceNet, only needed by the 4-dim aug leaf
CKPT_DIR = "/Users/shukaihu/Code_Project_Local/AlphaBig2-Value/checkpoints"


def _load_aug(value_ckpt=f"{CKPT_DIR}/value_aug.pt", dom_ckpt=f"{CKPT_DIR}/dominance_best.pt",
              mp_ckpt=f"{CKPT_DIR}/VALUE_minplays.pt"):
    """Load the augmented value leaves.
      value_mp  (extra_dim=1, VALUE+min-plays) — the deploy value (ablation: min-plays
                earns the whole +8.5% MSE; dominance is ~redundant for position value).
      value_aug (extra_dim=4, +dominance probs) — kept for reproducibility / A/B."""
    global _value_aug, _value_mp, _dom
    _value_mp = ValueNet(extra_dim=1).to(DEV)
    _value_mp.load_state_dict(torch.load(mp_ckpt, map_location=DEV)["model"]); _value_mp.eval()
    _value_aug = ValueNet(extra_dim=4).to(DEV)
    _value_aug.load_state_dict(torch.load(value_ckpt, map_location=DEV)["model"]); _value_aug.eval()
    _dom = DominanceNet().to(DEV)
    _dom.load_state_dict(torch.load(dom_ckpt, map_location=DEV)["model"]); _dom.eval()


@torch.no_grad()
def _policy_eval(env):
    legal, feats = legal_action_data(env)
    obs_t = _to_batch([obs_from_env(env)], DEV)
    af = torch.from_numpy(feats).unsqueeze(0).to(DEV)
    am = torch.ones(1, len(legal), dtype=torch.bool, device=DEV)
    logits, _ = _policy.forward(obs_t, af, am)
    return np.asarray(legal), torch.softmax(logits[0], -1).cpu().numpy()


@torch.no_grad()
def _greedy(env):
    legal, probs = _policy_eval(env)
    return int(legal[int(probs.argmax())])


@torch.no_grad()
def _leaf_value(env):
    """Normalized (tanh) expected final score for the to-move player."""
    return float(_value.forward(_to_batch([obs_from_env(env)], DEV))[0].item())


@torch.no_grad()
def _leaf_value_mp(env):
    """Deploy value: public obs + [min_plays/13] (the feature that earns the gain)."""
    b = _to_batch([obs_from_env(env)], DEV)
    mp = min_plays_to_empty([int(c) for c in env.game.currentHands[env.current_player]]) / 13.0
    b["feat"] = torch.tensor([[mp]], dtype=torch.float32, device=DEV)
    return float(_value_mp.forward(b)[0].item())


@torch.no_grad()
def _leaf_value_aug(env):
    """Dominance-aware value: public obs + [P(single/pair/fh nuts), min_plays/13]."""
    b = _to_batch([obs_from_env(env)], DEV)
    dp = torch.sigmoid(_dom.forward(b))[0]                       # (3,) strength probs
    me = env.current_player
    mp = min_plays_to_empty([int(c) for c in env.game.currentHands[me]]) / 13.0
    b["feat"] = torch.tensor([[float(dp[0]), float(dp[1]), float(dp[2]), mp]],
                             dtype=torch.float32, device=DEV)
    return float(_value_aug.forward(b)[0].item())


@torch.no_grad()
def _leaf_eval(env, me):
    """Leaf value (normalized, me's perspective).
      _LEAF='rollout'       → greedy-play to terminal (exact for the determinized world)
      _LEAF='value'         → VALUE.pt, optionally AFTER _VALUE_DEPTH greedy rounds of
                              lookahead (the more rounds, the fewer hidden cards remain,
                              so the value is less re-averaged / more world-specific;
                              _VALUE_DEPTH=∞ ≡ rollout). env is the leaf (our turn);
                              it's discarded after backup so we mutate it in place."""
    if _LEAF == "rollout":
        steps = 0
        while not env.done and steps < 200:
            env.step(_greedy(env)); steps += 1
        return float(np.tanh(env.game.rewards[me - 1] / SCALE))
    r = 0
    while r < _VALUE_DEPTH and not env.done:
        env.step(_greedy(env))                         # our greedy move
        while not env.done and env.current_player != me:
            env.step(_greedy(env))                     # opponents back to our turn
        r += 1
    if env.done:
        return float(np.tanh(env.game.rewards[me - 1] / SCALE))
    if _LEAF == "value_mp":
        return _leaf_value_mp(env)
    if _LEAF == "value_aug":
        return _leaf_value_aug(env)
    return _leaf_value(env)


def _determinize(env, me, belief):
    """Assign unseen cards to the 3 opponents (clockwise from me), exact counts.
    belief != None → importance-sample from belief; belief is None → uniform random."""
    g = env.game
    my = set(int(c) for c in g.currentHands[me])
    played = set(int(i) + 1 for i in np.flatnonzero(g.cardsPlayed.any(axis=0)))
    unseen = [c for c in range(1, 53) if c not in my and c not in played]
    np.random.shuffle(unseen)
    others = [((me - 1 + k) % 4) + 1 for k in (1, 2, 3)]   # matches belief row order
    rem = {p: int(g.currentHands[p].size) for p in others}
    out = {p: [] for p in others}
    for c in unseen:
        ps = [p for p in others if rem[p] > 0]
        if not ps:
            break
        if belief is None:
            p = ps[int(np.random.randint(len(ps)))]
        else:
            w = np.array([belief[others.index(p)][c - 1] for p in ps], dtype=np.float64) + 1e-6
            w /= w.sum()
            p = ps[int(np.random.choice(len(ps), p=w))]
        out[p].append(c); rem[p] -= 1
    return out


def _make_world(env, world):
    e = env.clone()
    for p, cards in world.items():
        e.game.currentHands[p] = np.sort(np.array(cards, dtype=np.int64))
    return e


def _simulate(env, tree, me, cpuct, root_noise=None):
    """root_noise: optional {action_idx: dirichlet_sample} — mixed into the ROOT prior only
    (_ROOT_NOISE). Root also gets a minimum-visit floor before PUCT free-selection (_MIN_VISITS)."""
    key = (); path = []; value = 0.0
    while True:
        if env.done:
            value = float(np.tanh(env.game.rewards[me - 1] / SCALE)); break
        if env.current_player != me:
            env.step(_greedy(env)); continue                 # opponent = environment (PPO_V4)
        legal_now, probs = _policy_eval(env)
        if key not in tree:                                  # leaf → expand + value eval
            tree[key] = {"N": {}, "W": {}, "avail": {}}
            value = _leaf_eval(env, me); break
        node = tree[key]
        is_root = not key
        if is_root and root_noise is not None:
            probs = np.array([(1 - _ROOT_NOISE_EPS) * float(probs[j])
                               + _ROOT_NOISE_EPS * root_noise.get(int(a), 0.0)
                               for j, a in enumerate(legal_now)], dtype=np.float32)
        if is_root and _MIN_VISITS > 0:
            starved = [int(a) for a in legal_now if node["N"].get(int(a), 0) < _MIN_VISITS]
            if starved:
                best_a = min(starved, key=lambda a: node["N"].get(a, 0))
                for a in legal_now:
                    a = int(a); node["avail"][a] = node["avail"].get(a, 0) + 1
                path.append((key, best_a)); env.step(best_a); key = key + (best_a,)
                continue
        sqrt_av = max(sum(node["avail"].get(int(a), 0) for a in legal_now), 1) ** 0.5
        best_a, best_s = int(legal_now[0]), -1e18
        for j, a in enumerate(legal_now):
            a = int(a); N = node["N"].get(a, 0)
            Q = (node["W"][a] / N) if N > 0 else 0.0
            s = Q + cpuct * float(probs[j]) * sqrt_av / (1 + N)
            if s > best_s:
                best_s, best_a = s, a
        for a in legal_now:
            a = int(a); node["avail"][a] = node["avail"].get(a, 0) + 1
        path.append((key, best_a)); env.step(best_a); key = key + (best_a,)
    for k, a in path:
        nd = tree[k]
        nd["N"][a] = nd["N"].get(a, 0) + 1
        nd["W"][a] = nd["W"].get(a, 0.0) + value


@torch.no_grad()
def alpha1_action(env, me, sims, cpuct, return_visits=False):
    legal, probs = _policy_eval(env)
    if len(legal) == 1:
        a = int(legal[0])
        return (a, {a: 1}) if return_visits else a
    # Root noise is sampled ONCE per decision (not per simulation): our own legal set + prior are
    # identical across all determinized worlds for this decision (they depend only on our public
    # state), so one noise vector consistently nudges exploration toward the same actions across
    # every world re-sample of this one search — matching AlphaZero's per-move root-noise cadence.
    root_noise = None
    if _ROOT_NOISE:
        noise = np.random.dirichlet([_ROOT_NOISE_ALPHA] * len(legal))
        root_noise = {int(a): float(n) for a, n in zip(legal, noise)}
    # _BELIEF_OVERRIDE(env) -> (3,52) belief array, letting an external harness
    # swap the belief source (e.g. history belief) without touching the ISMCTS.
    # None (default) = original behavior exactly.
    if _BELIEF_OVERRIDE is not None:
        belief = _BELIEF_OVERRIDE(env)
    else:
        belief = belief_from_obs(_belief, obs_from_env(env), DEV) if _USE_BELIEF else None
    tree = {}
    for _ in range(sims):
        try:
            _simulate(_make_world(env, _determinize(env, me, belief)), tree, me, cpuct, root_noise)
        except Exception:
            pass
    root = tree.get((), {"N": {}})
    best = max(root["N"], key=root["N"].get) if root["N"] else _greedy(env)
    _OVR[1] += 1
    if int(best) != int(_greedy(env)):   # did the search override the policy's greedy action?
        _OVR[0] += 1
    return (best, dict(root["N"])) if return_visits else best


def play_game(alpha_seat, sims, cpuct):
    env = Big2Env(); env.reset()
    steps = 0
    while not env.done and steps < 300:
        p = env.current_player
        a = alpha1_action(env, p, sims, cpuct) if p == alpha_seat else _greedy(env)
        env.step(a); steps += 1
    return env.game.rewards.copy()


def measure_disagreement(n_decisions=40, k_worlds=10, use_search=False, sims=80, cpuct=1.5,
                          seed=0, clear_gap=0.15, verbose=True):
    """Does the policy's top pick (or, if use_search, the ISMCTS's own chosen action) match the
    best action by REALIZED rollout outcome? For each of n_decisions real multi-option decisions,
    sample k_worlds ONCE (shared across candidates -> low-variance, common-random-numbers), and for
    each candidate action average its rollout-to-terminal outcome over those worlds. Compares:
      - policy top   = argmax of policy probs (no search)
      - search top   = alpha1_action's own chosen action (uses whatever _ROOT_NOISE/_MIN_VISITS/
                       _LEAF/_USE_BELIEF are currently set to — this is how Stage 1's de-anchoring
                       fix is validated: same 40 decisions, before vs after enabling the flags)
    against rollout top (best average realized outcome — the ground truth this diagnostic trusts).

    Returns a dict: agreement rates, a 'clear miss' rate (gap >= clear_gap, i.e. not noise-level
    given only k_worlds samples), and per-decision detail for inspection.
    """
    np.random.seed(seed)
    results = []
    env = Big2Env(); env.reset()
    while len(results) < n_decisions:
        me = env.current_player
        legal, probs = _policy_eval(env)
        if len(legal) > 1:
            belief = belief_from_obs(_belief, obs_from_env(env), DEV) if _USE_BELIEF else None
            worlds = [_determinize(env, me, belief) for _ in range(k_worlds)]
            pol_top = int(legal[int(np.argmax(probs))])
            search_top = alpha1_action(env, me, sims, cpuct) if use_search else None

            roll_avg = {}
            for a in legal:
                a = int(a); rvals = []
                for w in worlds:
                    e = _make_world(env, w); e.step(a)
                    steps = 0
                    while not e.done and steps < 200:
                        e.step(_greedy(e)); steps += 1
                    rvals.append(float(np.tanh(e.game.rewards[me - 1] / SCALE)))
                roll_avg[a] = float(np.mean(rvals))
            roll_top = max(roll_avg, key=roll_avg.get)
            cmp_top = search_top if use_search else pol_top
            gap = roll_avg[roll_top] - roll_avg[cmp_top]
            results.append({"n_legal": int(len(legal)), "cmp_top": cmp_top, "roll_top": roll_top,
                             "agree": cmp_top == roll_top, "gap": gap, "roll_avg": roll_avg})
            if verbose:
                print(f"  decision {len(results)}/{n_decisions}")

        env.step(_greedy(env))
        if env.done:
            env = Big2Env(); env.reset()

    n = len(results)
    agree = sum(r["agree"] for r in results)
    clear_miss = sum((not r["agree"]) and r["gap"] >= clear_gap for r in results)
    out = {"n": n, "agree_rate": agree / n, "clear_miss_rate": clear_miss / n,
           "use_search": use_search, "results": results}
    if verbose:
        tag = "search" if use_search else "policy"
        print(f"=== measure_disagreement (n={n}, k={k_worlds}, cmp={tag}) ===")
        print(f"  agree with rollout top : {agree}/{n} = {100*out['agree_rate']:.0f}%")
        print(f"  clear miss (gap>={clear_gap}): {clear_miss}/{n} = {100*out['clear_miss_rate']:.0f}%")
    return out


def precompute_ground_truth(n_decisions=40, k_worlds=10, seed=0, verbose=True):
    """Same rollout-ground-truth methodology as measure_disagreement, but CACHED so many different
    search configs can be swept against the IDENTICAL decisions + ground truth (removes the RNG-
    stream drift between configs that a fresh measure_disagreement() call per-config would have,
    and skips recomputing the expensive k_worlds rollouts every time)."""
    np.random.seed(seed)
    cached = []
    env = Big2Env(); env.reset()
    while len(cached) < n_decisions:
        me = env.current_player
        legal, probs = _policy_eval(env)
        if len(legal) > 1:
            belief = belief_from_obs(_belief, obs_from_env(env), DEV) if _USE_BELIEF else None
            worlds = [_determinize(env, me, belief) for _ in range(k_worlds)]
            roll_avg = {}
            for a in legal:
                a = int(a); rvals = []
                for w in worlds:
                    e = _make_world(env, w); e.step(a)
                    steps = 0
                    while not e.done and steps < 200:
                        e.step(_greedy(e)); steps += 1
                    rvals.append(float(np.tanh(e.game.rewards[me - 1] / SCALE)))
                roll_avg[a] = float(np.mean(rvals))
            roll_top = max(roll_avg, key=roll_avg.get)
            cached.append({"env": env.clone(), "me": me,
                           "pol_top": int(legal[int(np.argmax(probs))]),
                           "roll_avg": roll_avg, "roll_top": roll_top})
            if verbose:
                print(f"  ground truth {len(cached)}/{n_decisions}")
        env.step(_greedy(env))
        if env.done:
            env = Big2Env(); env.reset()
    return cached


def evaluate_against_ground_truth(cached, sims=80, cpuct=1.5, use_search=True,
                                   clear_gap=0.15, verbose=True, tag=""):
    """Sweep helper: compare policy-top (use_search=False) or alpha1_action's chosen action
    (use_search=True, using whatever _ROOT_NOISE/_MIN_VISITS/_LEAF are CURRENTLY set) against
    `cached` ground truth from precompute_ground_truth(). Cheap — no rollout recomputation."""
    agree, clear_miss = 0, 0
    for d in cached:
        cmp_top = alpha1_action(d["env"].clone(), d["me"], sims, cpuct) if use_search else d["pol_top"]
        is_agree = cmp_top == d["roll_top"]
        agree += is_agree
        gap = d["roll_avg"][d["roll_top"]] - d["roll_avg"][cmp_top]
        if not is_agree and gap >= clear_gap:
            clear_miss += 1
    n = len(cached)
    out = {"n": n, "agree_rate": agree / n, "clear_miss_rate": clear_miss / n}
    if verbose:
        print(f"  [{tag}] agree {agree}/{n}={100*out['agree_rate']:.0f}%  "
              f"clear-miss {clear_miss}/{n}={100*out['clear_miss_rate']:.0f}%")
    return out


def main():
    global _LEAF, _VALUE_DEPTH, _USE_BELIEF, _ROOT_NOISE, _ROOT_NOISE_EPS, _ROOT_NOISE_ALPHA, _MIN_VISITS
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--cpuct", type=float, default=1.5)
    ap.add_argument("--leaf", choices=["value", "value_mp", "value_aug", "rollout"], default="value",
                    help="value_mp = deploy value (VALUE + min-plays)")
    ap.add_argument("--value-depth", type=int, default=0, help="greedy rounds of lookahead before VALUE.pt")
    ap.add_argument("--belief", choices=["on", "off"], default="on", help="off = uniform determinization")
    ap.add_argument("--paired", action="store_true", help="same-deal paired comparison (cancels deal luck)")
    ap.add_argument("--value-ckpt", default="", help="override the value net checkpoint")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root-noise", action="store_true", help="de-anchor: Dirichlet noise at root (default off)")
    ap.add_argument("--root-noise-eps", type=float, default=_ROOT_NOISE_EPS)
    ap.add_argument("--root-noise-alpha", type=float, default=_ROOT_NOISE_ALPHA)
    ap.add_argument("--min-visits", type=int, default=0, help="de-anchor: force >=N visits/root action before PUCT (0=off)")
    ap.add_argument("--measure-disagreement", action="store_true",
                    help="run measure_disagreement() instead of a game match (Stage 1 validation)")
    ap.add_argument("--use-search", action="store_true",
                    help="with --measure-disagreement: compare the SEARCH's chosen action (not raw policy) vs rollout ground truth")
    ap.add_argument("--k-worlds", type=int, default=10, help="measure_disagreement: worlds/candidate action")
    args = ap.parse_args()
    _LEAF = args.leaf; _VALUE_DEPTH = args.value_depth; _USE_BELIEF = (args.belief == "on")
    _ROOT_NOISE = args.root_noise; _ROOT_NOISE_EPS = args.root_noise_eps
    _ROOT_NOISE_ALPHA = args.root_noise_alpha; _MIN_VISITS = args.min_visits
    if args.leaf in ("value_mp", "value_aug"):
        _load_aug()
    if args.value_ckpt:
        _value.load_state_dict(torch.load(args.value_ckpt, map_location=DEV)["model"])
        print(f"value net <- {os.path.basename(args.value_ckpt)}")
    torch.manual_seed(args.seed)
    cfg = f"leaf={args.leaf} vdepth={args.value_depth} belief={args.belief}"

    if args.measure_disagreement:
        print(f"de-anchor: root_noise={_ROOT_NOISE}(eps={_ROOT_NOISE_EPS},alpha={_ROOT_NOISE_ALPHA}) "
              f"min_visits={_MIN_VISITS} | cmp={'search' if args.use_search else 'policy'}")
        measure_disagreement(n_decisions=args.games, k_worlds=args.k_worlds, use_search=args.use_search,
                              sims=args.sims, cpuct=args.cpuct, seed=args.seed)
        return

    if args.paired:
        # Common random numbers: each deal is played TWICE — once with Alpha1 at the
        # seat, once with plain V4 at the same seat (everyone else V4 both times) — on
        # the SAME cards. Re-seeding np.random before each play makes env.reset() deal
        # the identical hands, so deal luck cancels in the per-deal difference.
        print(f"PAIRED Alpha1({cfg}) vs V4 | {args.games} deals (same cards)")
        ds = []
        for i in range(args.games):
            seat = (i % 4) + 1
            np.random.seed(args.seed * 100000 + i); rA = play_game(seat, args.sims, args.cpuct)
            np.random.seed(args.seed * 100000 + i); rB = play_game(0, args.sims, args.cpuct)
            ds.append(float(rA[seat - 1] - rB[seat - 1]))
            if (i + 1) % 10 == 0:
                m = np.mean(ds); se = np.std(ds, ddof=1) / np.sqrt(len(ds))
                print(f"  {i+1}/{args.games} | Alpha1−V4 (paired) {m:+.2f} ± {se:.2f}")
        m = np.mean(ds); se = np.std(ds, ddof=1) / np.sqrt(len(ds))
        print(f"\n=== PAIRED RESULT ({len(ds)} deals) ===")
        print(f"Alpha1 − V4 (same deal): {m:+.2f} ± {se:.2f}/game   t={m/max(se,1e-9):+.1f}")
        print(f"paired std {np.std(ds, ddof=1):.1f} (vs unpaired ~23 → ~{23/max(np.std(ds,ddof=1),0.1):.0f}x variance reduction)")
        return

    np.random.seed(args.seed)
    print(f"Alpha1 (ISMCTS sims={args.sims} {cfg}) vs V4 | {args.games} games")

    a_scores, v_scores, a_wins = [], [], 0
    t0 = time.time()
    for i in range(args.games):
        seat = (i % 4) + 1                                   # rotate Alpha1's seat
        r = play_game(seat, args.sims, args.cpuct)
        a_scores.append(float(r[seat - 1]))
        v_scores.extend(float(r[p - 1]) for p in range(1, 5) if p != seat)
        if int(np.argmax(r)) == seat - 1:
            a_wins += 1
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{args.games} | Alpha1 avg {np.mean(a_scores):+.2f} | "
                  f"V4 avg {np.mean(v_scores):+.2f} | {(time.time()-t0)/(i+1):.1f}s/game")
    print("\n=== RESULT ===")
    print(f"Alpha1: avg {np.mean(a_scores):+.2f}  (per game, {len(a_scores)} games)  win {100*a_wins/args.games:.0f}%")
    print(f"V4    : avg {np.mean(v_scores):+.2f}  (per seat, {len(v_scores)} seat-games)")
    print(f"Alpha1 − V4 margin: {np.mean(a_scores) - np.mean(v_scores):+.2f} / game")
    print(f"(Alpha1 = V4 + ISMCTS; positive margin = the search helps. 4-player chance win rate = 25%.)")
    print(f"search OVERRODE greedy action: {_OVR[0]}/{_OVR[1]} = {100*_OVR[0]/max(_OVR[1],1):.1f}% of decisions")


if __name__ == "__main__":
    main()
