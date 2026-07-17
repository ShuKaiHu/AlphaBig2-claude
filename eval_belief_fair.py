"""Fair belief comparison: BELIEF.pt vs belief_1266 on a TEMPORALLY held-out
test set -- real games added to online_games.jsonl AFTER the exact 1266-game
snapshot both checkpoints were trained on (ppo/data/snapshot_1266_ids.json).

This is a genuinely fair test (unlike the random 10% split each training run
draws from whatever's currently in the corpus): NEITHER checkpoint has ever
seen these games during training, and they're temporally separated from the
training pool (not just a different random shuffle of the same pool), so the
comparison isn't confounded by "which random split did each training run
happen to get".

    ./.venv/bin/python eval_belief_fair.py
"""
import json
import os

import numpy as np
import torch

from ppo.bc_dataset import _snapshot
from ppo.belief_model import BeliefNet, belief_batch, belief_bce
from ppo.belief_history_dataset import build_belief_history_records
from ppo.belief_model_history import BeliefNetHistory, belief_history_batch
from ppo.network_cardaware import ACTION_FEATURES, obs_from_env
from ppo.network_v6 import CardAwareV6, belief_target_from_env, build_batch_v6, unseen_mask_from_obs

DATA = os.path.join(os.path.dirname(__file__), "ppo", "data", "online_games.jsonl")
SNAPSHOT = os.path.join(os.path.dirname(__file__), "ppo", "data", "snapshot_1266_ids.json")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "ppo", "checkpoints")

PHASES = [("early <13", 0, 13), ("mid 13-25", 13, 26), ("late 26-38", 26, 39), ("end 39+", 39, 99)]


class _Shim:
    __slots__ = ("game",)
    def __init__(self, g): self.game = g
    @property
    def current_player(self): return self.game.playersGo


def game_content_id(g):
    import hashlib
    core = json.dumps({"hands": g["hands"], "plays": g["plays"]}, sort_keys=True)
    return hashlib.md5(core.encode()).hexdigest()[:16]


def build_belief_records(games, target="all"):
    import enumerateOptions as eo
    records = []
    for g in games:
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        our = g["our_seat"]; winner = g["winner"]
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        trick_cards, trick_owner, passed = None, None, set()
        for ev in g["plays"]:
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            control = 1 if trick_cards is None else 0
            want = (target == "all" or (target == "human" and s != our)
                    or (target == "winner" and s == winner)
                    or (target == "ours" and s == our))
            if want:
                remaining = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
                gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                try:
                    mask = gg.returnAvailableActions()
                    legal = np.flatnonzero(mask)
                    label = eo.passInd if act == "pass" else eo.action_index_from_cards(cards)
                    pos = int(np.where(legal == label)[0][0]) if label in legal else -1
                except Exception:
                    pos = -1
                if pos >= 0:
                    shim = _Shim(gg)
                    obs = obs_from_env(shim)
                    records.append({"obs": obs,
                                    "feats": ACTION_FEATURES[legal].astype(np.float32),
                                    "pos": pos,
                                    "belief": belief_target_from_env(shim),
                                    "bmask": unseen_mask_from_obs(obs)})
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cardsPlayed[s][c - 1] = 1
                trick_cards, trick_owner, passed = cards, s, set()
            else:
                passed.add(s)
                if len(passed) >= 3:
                    trick_cards, trick_owner, passed = None, None, set()
    return records


@torch.no_grad()
def metrics(net, recs, device):
    net.eval()
    ps = pn = base = bn = bce = 0.0
    ph = {p[0]: [0.0, 0.0, 0] for p in PHASES}  # hit_sum, base_sum, n
    ii = np.arange(len(recs))
    for s in range(0, len(ii), 512):
        mb = [recs[i] for i in ii[s:s + 512]]
        b = belief_batch(mb, device)
        logits = net(b["obs"])
        bce += belief_bce(logits, b["belief"], b["bmask"]).item() * len(mb)
        P = (torch.sigmoid(logits) * b["bmask"].unsqueeze(1)).cpu().numpy()
        T = b["belief"].cpu().numpy(); M = b["bmask"].cpu().numpy()
        for bi in range(P.shape[0]):
            nun = int(M[bi].sum())
            played = int(round(float(mb[bi]["obs"]["seen"].sum())))
            name = next(p[0] for p in PHASES if p[1] <= played < p[2])
            for o in range(3):
                k = int(T[bi, o].sum())
                if k == 0:
                    continue
                top = np.argpartition(-P[bi, o], k - 1)[:k]
                hit = float(T[bi, o, top].sum()) / k
                b_o = k / max(nun, 1)
                ps += hit; pn += 1
                base += b_o; bn += 1
                ph[name][0] += hit; ph[name][1] += b_o; ph[name][2] += 1
    return (ps / max(pn, 1), base / max(bn, 1), bce / max(len(recs), 1),
            {n: (v[0] / v[2] if v[2] else 0.0, v[1] / v[2] if v[2] else 0.0, v[2]) for n, v in ph.items()})


@torch.no_grad()
def metrics_v6(net, recs, device):
    """Same P@count/BCE metric as `metrics()`, but for CardAwareV6 -- its belief
    head is baked into the joint policy/value forward pass (needs act_feats/
    act_mask too), not a standalone BeliefNet, so the batch/forward call differs
    but the belief_logits shape (B,3,52) and semantics are identical."""
    net.eval()
    ps = pn = base = bn = bce = 0.0
    ph = {p[0]: [0.0, 0.0, 0] for p in PHASES}  # hit_sum, base_sum, n
    ii = np.arange(len(recs))
    for s in range(0, len(ii), 512):
        mb = [recs[i] for i in ii[s:s + 512]]
        b = build_batch_v6(mb, device)
        _, _, logits = net.forward(b["obs"], b["act_feats"], b["act_mask"])
        bce += belief_bce(logits, b["belief"], b["bmask"]).item() * len(mb)
        P = (torch.sigmoid(logits) * b["bmask"].unsqueeze(1)).cpu().numpy()
        T = b["belief"].cpu().numpy(); M = b["bmask"].cpu().numpy()
        for bi in range(P.shape[0]):
            nun = int(M[bi].sum())
            played = int(round(float(mb[bi]["obs"]["seen"].sum())))
            name = next(p[0] for p in PHASES if p[1] <= played < p[2])
            for o in range(3):
                k = int(T[bi, o].sum())
                if k == 0:
                    continue
                top = np.argpartition(-P[bi, o], k - 1)[:k]
                hit = float(T[bi, o, top].sum()) / k
                b_o = k / max(nun, 1)
                ps += hit; pn += 1
                base += b_o; bn += 1
                ph[name][0] += hit; ph[name][1] += b_o; ph[name][2] += 1
    return (ps / max(pn, 1), base / max(bn, 1), bce / max(len(recs), 1),
            {n: (v[0] / v[2] if v[2] else 0.0, v[1] / v[2] if v[2] else 0.0, v[2]) for n, v in ph.items()})


def build_belief_records_v1(games, target="all"):
    """V1 schema records (HIST_STEP_DIM=29, reuses the OLD engine.features.encode_history_steps)
    -- ONLY for evaluating the old belief_history_578/1266 checkpoints (trained before the
    redesign to raw-card required/played multi-hots) as a before/after comparison. Does not
    touch engine/features.py, just consumes it read-only, same as its other existing callers."""
    import enumerateOptions as eo
    from engine.features import HISTORY_LEN, encode_history_steps

    class _HistShimV1:
        __slots__ = ("actionHistory",)
        def __init__(self, history): self.actionHistory = history

    records = []
    for g in games:
        hands = {int(s): list(map(int, g["hands"][s])) for s in g["hands"]}
        our = g["our_seat"]; winner = g["winner"]
        played = {s: [] for s in range(4)}
        cardsPlayed = np.zeros((4, 52), dtype=np.int64)
        trick_cards, trick_owner, passed = None, None, set()
        action_history = []
        for ev in g["plays"]:
            s = ev["seat"]; act = ev["action"]; cards = list(map(int, ev["cards"]))
            control = 1 if trick_cards is None else 0
            want = (target == "all" or (target == "human" and s != our)
                    or (target == "winner" and s == winner)
                    or (target == "ours" and s == our))
            if want:
                remaining = {k: [c for c in hands[k] if c not in set(played[k])] for k in range(4)}
                gg = _snapshot(s, remaining, cardsPlayed, control, trick_cards, trick_owner, passed)
                try:
                    mask = gg.returnAvailableActions()
                    legal = np.flatnonzero(mask)
                    label = eo.passInd if act == "pass" else eo.action_index_from_cards(cards)
                    pos = int(np.where(legal == label)[0][0]) if label in legal else -1
                except Exception:
                    pos = -1
                if pos >= 0:
                    shim = _Shim(gg)
                    obs = obs_from_env(shim)
                    hist = encode_history_steps(_HistShimV1(action_history), history_len=HISTORY_LEN)
                    records.append({"obs": obs,
                                    "belief": belief_target_from_env(shim),
                                    "bmask": unseen_mask_from_obs(obs),
                                    "history": hist.astype(np.float32)})
            action_history.append({
                "player": s + 1,
                "hand": np.array(cards, dtype=np.int64) if act == "play" else None,
                "pass": act == "pass",
                "forced_skip": False,
            })
            if act == "play":
                played[s].extend(cards)
                for c in cards:
                    cardsPlayed[s][c - 1] = 1
                trick_cards, trick_owner, passed = cards, s, set()
            else:
                passed.add(s)
                if len(passed) >= 3:
                    trick_cards, trick_owner, passed = None, None, set()
    return records


@torch.no_grad()
def metrics_history(net, recs, device):
    """Same P@count/BCE metric, for BeliefNetHistory (either V1 29-dim or V2 108-dim schema --
    whichever the records were built with; the net's GRU input_size must match)."""
    net.eval()
    ps = pn = base = bn = bce = 0.0
    ph = {p[0]: [0.0, 0.0, 0] for p in PHASES}
    ii = np.arange(len(recs))
    for s in range(0, len(ii), 256):
        mb = [recs[i] for i in ii[s:s + 256]]
        b = belief_history_batch(mb, device)
        logits = net(b["obs"], b["history"])
        bce += belief_bce(logits, b["belief"], b["bmask"]).item() * len(mb)
        P = (torch.sigmoid(logits) * b["bmask"].unsqueeze(1)).cpu().numpy()
        T = b["belief"].cpu().numpy(); M = b["bmask"].cpu().numpy()
        for bi in range(P.shape[0]):
            nun = int(M[bi].sum())
            played = int(round(float(mb[bi]["obs"]["seen"].sum())))
            name = next(p[0] for p in PHASES if p[1] <= played < p[2])
            for o in range(3):
                k = int(T[bi, o].sum())
                if k == 0:
                    continue
                top = np.argpartition(-P[bi, o], k - 1)[:k]
                hit = float(T[bi, o, top].sum()) / k
                b_o = k / max(nun, 1)
                ps += hit; pn += 1
                base += b_o; bn += 1
                ph[name][0] += hit; ph[name][1] += b_o; ph[name][2] += 1
    return (ps / max(pn, 1), base / max(bn, 1), bce / max(len(recs), 1),
            {n: (v[0] / v[2] if v[2] else 0.0, v[1] / v[2] if v[2] else 0.0, v[2]) for n, v in ph.items()})


def _print_result(tag, pc, base, bce, ph):
    phs = "  ".join(f"{n.split()[0]} {v[0]*100:.1f}%/base{v[1]*100:.1f}%/lift{(v[0]-v[1])*100:+.1f}(n={v[2]})"
                    for n, v in ph.items())
    print(f"\n{tag}")
    print(f"  P@count: {pc*100:.1f}% (baseline {base*100:.1f}%) bce={bce:.3f}")
    print(f"  by phase: {phs}")


def main():
    snapshot_ids = set(json.load(open(SNAPSHOT)))
    all_games = [json.loads(l) for l in open(DATA)]
    new_games = [g for g in all_games if (g.get("id") or game_content_id(g)) not in snapshot_ids]

    print(f"total games in corpus: {len(all_games)}")
    print(f"games in training snapshot (seen by both checkpoints): {len(snapshot_ids)}")
    print(f"NEW held-out games (unseen by either checkpoint): {len(new_games)}")
    if not new_games:
        print("no new games yet -- nothing to evaluate. Run more online tests first.")
        return

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    recs = build_belief_records(new_games, target="all")
    print(f"held-out decision records: {len(recs)}")
    if not recs:
        print("no usable decision records extracted from the new games.")
        return

    for tag, path in [("BELIEF.pt (original, ~578-game era)", os.path.join(CKPT_DIR, "saved", "BELIEF.pt")),
                       ("belief_1266 (1266-game retrain)", os.path.join(CKPT_DIR, "belief_1266_best.pt"))]:
        ck = torch.load(path, map_location=device, weights_only=False)
        net = BeliefNet().to(device)
        net.load_state_dict(ck["model"])
        _print_result(tag, *metrics(net, recs, device))

    v6_path = os.path.join(CKPT_DIR, "ppo_v6_human_belief.pt")
    if os.path.exists(v6_path):
        ck = torch.load(v6_path, map_location=device, weights_only=False)
        net = CardAwareV6().to(device)
        net.load_state_dict(ck["model"])
        _print_result("ppo_v6_human_belief.pt (V6 architecture, belief head baked into policy/value net)",
                       *metrics_v6(net, recs, device))

    # V1-schema belief_history checkpoints (old encode_history_steps reuse, HIST_STEP_DIM=29) --
    # kept around purely for a before/after comparison against the V2 (raw-card) redesign.
    v1_paths = [("belief_history_578 (V1 schema, 578-game era)", os.path.join(CKPT_DIR, "belief_history_578_best.pt")),
                ("belief_history_1266 (V1 schema, 1266-game era)", os.path.join(CKPT_DIR, "belief_history_1266_best.pt"))]
    if any(os.path.exists(p) for _, p in v1_paths):
        recs_v1 = build_belief_records_v1(new_games, target="all")
        for tag, path in v1_paths:
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=device, weights_only=False)
            net = BeliefNetHistory(hist_dim=29).to(device)  # old V1 schema dim
            net.load_state_dict(ck["model"])
            _print_result(tag, *metrics_history(net, recs_v1, device))

    # V2-schema belief_history checkpoints (raw required/played card multi-hots, HIST_STEP_DIM_V2=108)
    v2_paths = [("belief_history_v2_578 (V2 schema, 578-game era)", os.path.join(CKPT_DIR, "belief_history_v2_578_best.pt")),
                ("belief_history_v2_1266 (V2 schema, 1266-game era)", os.path.join(CKPT_DIR, "belief_history_v2_1266_best.pt"))]
    if any(os.path.exists(p) for _, p in v2_paths):
        recs_v2 = build_belief_history_records(new_games, target="all")
        for tag, path in v2_paths:
            if not os.path.exists(path):
                continue
            ck = torch.load(path, map_location=device, weights_only=False)
            net = BeliefNetHistory().to(device)
            net.load_state_dict(ck["model"])
            _print_result(tag, *metrics_history(net, recs_v2, device))


if __name__ == "__main__":
    main()
