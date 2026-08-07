"""Web UI backend — play Big 2 against the engine checkpoints in a browser.

The browser mirror of ppo/play_human.py, but on the engine (Big2Net + MCTS)
stack whose deploy checkpoints actually live in this repo
(engine/checkpoints/saved/*.pt). One Flask process, in-memory game sessions.

Run from the repo root:
    python -m web.server                    # http://127.0.0.1:8288
    python -m web.server --port 9000 --host 0.0.0.0

AI move modes (per game, chosen in the UI):
    greedy        — policy argmax, no search (fast; = evaluator._model_action)
    mcts_det      — determinized MCTS: the model does NOT see hidden hands;
                    it samples N determinizations of the unknown cards and sums
                    visit counts (= eval_determinization.py, the online-like
                    path). Uses the V9 full-info value net when the checkpoint
                    carries one (toggleable).
    mcts_fullinfo — MCTS on the TRUE hands. The AI is cheating; debug only,
                    clearly labeled in the UI.

Feature-dim handling: checkpoints span 302/306/312 static dims. We reconfigure
engine.features globals per loaded checkpoint (mirroring features.set_combo and
eval_reward.load_model's auto-detect), guarded by a lock so concurrent games on
different checkpoints can't interleave encodes with a wrong STATIC_DIM.

Finished games are appended to web/human_games_web.jsonl (same spirit as
ppo/play_human.py's log): per-seat scores, config, and the full action list.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory

import enumerateOptions as _eo
import gameLogic
from engine import features as _F
from engine.env import Big2Env, PASS_IDX
from engine.mcts import MCTS
from engine.model import Big2Net, Big2ValueNet

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(WEB_DIR)
CKPT_DIR = os.path.join(REPO_ROOT, "engine", "checkpoints", "saved")
LOG_PATH = os.path.join(WEB_DIR, "human_games_web.jsonl")

RANKS = ['3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2']
SUITS = ['♦', '♣', '♥', '♠']
SUIT_NAMES = ['d', 'c', 'h', 's']

app = Flask(__name__, static_folder=None)

# ── model registry ───────────────────────────────────────────────────────────

_model_lock = threading.Lock()   # serializes feature-global switching + inference
_model_cache = {}                # name -> dict(model, value_model, static_dim)


def list_checkpoints():
    out = []
    if os.path.isdir(CKPT_DIR):
        for fn in sorted(os.listdir(CKPT_DIR)):
            if fn.endswith(".pt"):
                out.append(fn)
    return out


def _set_feature_dims(static_dim: int):
    """Reconfigure engine.features globals for a checkpoint's static width.
    302 = no dominance; 306 = +dominance; 312 = +dominance +combo."""
    dominance_on = static_dim != 302
    _F.DOMINANCE_ON = dominance_on
    _F._DOMINANCE_DIM = 4 if dominance_on else 0
    _F.set_combo(static_dim >= 302 + 4 + 6)
    if _F.STATIC_DIM != static_dim:
        raise ValueError(f"unsupported static dim {static_dim} "
                         f"(reconfigured to {_F.STATIC_DIM})")


def load_checkpoint(name: str):
    """Load (cached) — caller must hold _model_lock. Also leaves the feature
    globals configured for this checkpoint."""
    if name in _model_cache:
        _set_feature_dims(_model_cache[name]["static_dim"])
        return _model_cache[name]
    path = os.path.join(CKPT_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        raise FileNotFoundError(name)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    static_dim = int(state["input_proj.0.weight"].shape[1]) - _F.GRU_HIDDEN
    _set_feature_dims(static_dim)
    model = Big2Net()
    model.load_state_dict(state)
    model.eval()
    value_model = None
    if isinstance(ckpt, dict) and "value_state" in ckpt:
        value_model = Big2ValueNet()
        value_model.load_state_dict(ckpt["value_state"])
        value_model.eval()
    entry = {"model": model, "value_model": value_model, "static_dim": static_dim}
    _model_cache[name] = entry
    return entry


# ── card / action helpers ────────────────────────────────────────────────────

def card_obj(cid):
    cid = int(cid)
    return {"id": cid, "rank": RANKS[(cid - 1) // 4], "suit": SUIT_NAMES[(cid - 1) % 4],
            "str": f"{RANKS[(cid - 1) // 4]}{SUITS[(cid - 1) % 4]}"}


def cards_str(cards):
    return ' '.join(card_obj(c)["str"] for c in sorted(int(c) for c in cards))


def five_kind(cards):
    arr = np.array(sorted(int(c) for c in cards))
    if gameLogic.isStraightFlush(arr):
        return "同花順"
    if gameLogic.isFourOfAKind(arr):
        return "鐵支"
    if gameLogic.isFullHouse(arr)[0]:
        return "葫蘆"
    return "順子"


def action_obj(a):
    a = int(a)
    if a == PASS_IDX:
        return {"action": a, "type": "pass", "cards": [], "label": "PASS"}
    cards, nc = _eo.getOptionNC(a)
    cards = sorted(int(c) for c in cards)
    if nc == 1:
        t, label = "single", f"單張 {cards_str(cards)}"
    elif nc == 2:
        t, label = "pair", f"對子 {cards_str(cards)}"
    else:
        k = five_kind(cards)
        t, label = "five", f"{k} {cards_str(cards)}"
    return {"action": a, "type": t, "cards": [card_obj(c) for c in cards], "label": label}


# ── AI move logic (canonical paths) ──────────────────────────────────────────

def greedy_action(model, env):
    """= engine.evaluator._model_action."""
    game = env.game
    player = env.current_player
    static = _F.encode_static(game, player)
    history = _F.encode_history_steps(game)
    valid = env.get_valid_actions()
    probs, _ = model.predict(static, history, valid)
    masked = probs * valid
    if masked.sum() == 0:
        return PASS_IDX
    return int(np.argmax(masked))


def belief_probs(model, game, player):
    static = _F.encode_static(game, player)
    history = _F.encode_history_steps(game)
    sf = torch.FloatTensor(static).unsqueeze(0)
    hs = torch.FloatTensor(history).unsqueeze(0)
    with torch.no_grad():
        _, _, bl = model.forward(sf, hs, None)
    return torch.sigmoid(bl).squeeze(0).numpy().reshape(3, 52)


def sample_assignment(unknown, counts, others, belief=None, rng=None):
    """= eval_determinization.sample_assignment, generalized to any seat set."""
    rng = rng or np.random
    order = list(unknown)
    rng.shuffle(order)
    caps = dict(counts)
    out = {o: [] for o in others}
    for card in order:
        avail = [o for o in others if caps[o] > 0]
        if not avail:
            break
        if belief is not None:
            w = np.array([max(belief[others.index(o)][card - 1], 1e-6) for o in avail])
            w = w / w.sum()
            owner = avail[rng.choice(len(avail), p=w)]
        else:
            owner = avail[rng.choice(len(avail))]
        out[owner].append(card)
        caps[owner] -= 1
    return {o: np.sort(np.array(v, dtype=np.int64)) for o, v in out.items()}


def determinized_action(model, value_model, true_env, cfg):
    """= eval_determinization.determinized_action, generalized: any seat,
    optional V9 value net, configurable c_puct / dirichlet."""
    game = true_env.game
    me = true_env.current_player
    others = [p for p in range(1, 5) if p != me]
    my = set(int(c) for c in game.currentHands[me])
    all_cards = set()
    for p in range(1, 5):
        all_cards |= set(int(c) for c in game.currentHands[p])
    unknown = sorted(all_cards - my)
    counts = {p: int(game.currentHands[p].size) for p in others}

    def mk_mcts():
        return MCTS(model, n_simulations=cfg["sims"], c_puct=cfg["c_puct"],
                    dirichlet_frac=cfg["dirichlet"],
                    value_model=value_model if cfg["use_value_net"] else None)

    if len(unknown) == 0 or sum(counts.values()) == 0:
        _, visits = mk_mcts().run(true_env, temperature=1.0)
        return int(np.argmax(visits))

    belief = belief_probs(model, game, me) if cfg["belief"] else None
    agg = None
    for _ in range(cfg["dets"]):
        assign = sample_assignment(unknown, counts, others, belief=belief)
        d = true_env.clone()
        for p in others:
            d.game.currentHands[p] = assign[p]
        _, visits = mk_mcts().run(d, temperature=1.0)
        agg = visits if agg is None else agg + visits
    return int(np.argmax(agg))


def fullinfo_mcts_action(model, value_model, env, cfg):
    mcts = MCTS(model, n_simulations=cfg["sims"], c_puct=cfg["c_puct"],
                dirichlet_frac=cfg["dirichlet"],
                value_model=value_model if cfg["use_value_net"] else None)
    _, visits = mcts.run(env, temperature=1.0)
    return int(np.argmax(visits))


def ai_action(entry, env, cfg):
    t0 = time.time()
    if cfg["mode"] == "greedy":
        a = greedy_action(entry["model"], env)
    elif cfg["mode"] == "mcts_det":
        a = determinized_action(entry["model"], entry["value_model"], env, cfg)
    elif cfg["mode"] == "mcts_fullinfo":
        a = fullinfo_mcts_action(entry["model"], entry["value_model"], env, cfg)
    else:
        raise ValueError(f"unknown mode {cfg['mode']}")
    return a, time.time() - t0


# ── game sessions ────────────────────────────────────────────────────────────

_sessions = {}     # id -> session dict
_sessions_lock = threading.Lock()

DEFAULT_CFG = {
    "checkpoint": "v9c_combo_strongopp_deploy.pt",
    "mode": "mcts_det",
    "sims": 50,
    "dets": 4,
    "belief": True,
    "use_value_net": True,
    "c_puct": 2.0,
    "dirichlet": 0.0,
    "reveal": False,
    "log_ai_analysis": False,
}


def parse_cfg(body):
    cfg = dict(DEFAULT_CFG)
    for k in cfg:
        if k in body:
            cfg[k] = body[k]
    cfg["sims"] = max(1, min(int(cfg["sims"]), 1000))
    cfg["dets"] = max(1, min(int(cfg["dets"]), 32))
    cfg["c_puct"] = float(cfg["c_puct"])
    cfg["dirichlet"] = max(0.0, min(float(cfg["dirichlet"]), 1.0))
    for k in ("belief", "use_value_net", "reveal", "log_ai_analysis"):
        cfg[k] = bool(cfg[k])
    if cfg["mode"] not in ("greedy", "mcts_det", "mcts_fullinfo"):
        raise ValueError(f"bad mode {cfg['mode']}")
    return cfg


def new_session(body):
    cfg = parse_cfg(body)
    seed = body.get("seed")
    if seed is not None and str(seed) != "":
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    else:
        seed = None
    seat = body.get("human_seat", "random")
    if seat == "random":
        human_seat = int(np.random.randint(1, 5))
    else:
        human_seat = int(seat)
        if human_seat not in (1, 2, 3, 4):
            raise ValueError("human_seat must be 1-4 or 'random'")

    with _model_lock:
        load_checkpoint(cfg["checkpoint"])  # fail fast + warm cache

    env = Big2Env()
    env.reset()
    initial_hands = {p: [int(c) for c in env.game.currentHands[p]] for p in range(1, 5)}
    sid = uuid.uuid4().hex[:12]
    s = {
        "id": sid, "env": env, "cfg": cfg, "human_seat": human_seat,
        "seed": seed, "initial_hands": initial_hands,
        "actions": [],          # [(player, action_int, ai_time or None)]
        "ai_notes": [],         # per-AI-move analysis lines (optional)
        "rewards": None,
        "created": time.time(),
    }
    with _sessions_lock:
        _sessions[sid] = s
    return s


def get_session(sid):
    with _sessions_lock:
        s = _sessions.get(sid)
    if s is None:
        raise KeyError(sid)
    return s


def replay_to(s, n_actions):
    """Rebuild env from the recorded deal, replaying the first n_actions."""
    env = Big2Env()
    env.reset(hands=s["initial_hands"])
    for (player, a, _t) in s["actions"][:n_actions]:
        assert env.current_player == player, "replay desync"
        env.step(int(a))
    s["env"] = env
    s["actions"] = s["actions"][:n_actions]
    s["rewards"] = None


def apply_action(s, action, ai_time=None):
    env = s["env"]
    player = env.current_player
    rewards, done = env.step(int(action))
    s["actions"].append((player, int(action), ai_time))
    if done:
        s["rewards"] = [float(x) for x in rewards]
        _log_finished_game(s)
    return done


def _log_finished_game(s):
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps({
                "ts": time.time(), "session": s["id"],
                "human_seat": s["human_seat"], "seed": s["seed"],
                "scores": s["rewards"], "human_score": s["rewards"][s["human_seat"] - 1],
                "cfg": {k: v for k, v in s["cfg"].items() if k != "reveal"},
                "n_actions": len(s["actions"]),
                "actions": [[p, a] for (p, a, _t) in s["actions"]],
                "initial_hands": s["initial_hands"],
            }) + "\n")
    except OSError:
        pass


def advance_ai(s, max_moves=64):
    """Let AI seats act until it's the human's turn or the game ends.
    Returns list of AI move dicts."""
    env = s["env"]
    cfg = s["cfg"]
    moves = []
    with _model_lock:
        entry = load_checkpoint(cfg["checkpoint"])
        while (not env.done and env.current_player != s["human_seat"]
               and len(moves) < max_moves):
            p = env.current_player
            a, dt = ai_action(entry, env, cfg)
            note = None
            if cfg["log_ai_analysis"]:
                note = _quick_policy_note(entry["model"], env, a)
            apply_action(s, a, ai_time=dt)
            m = {"player": p, **action_obj(a), "time": round(dt, 3)}
            if note:
                m["policy_note"] = note
            moves.append(m)
    return moves


def _quick_policy_note(model, env, chosen):
    """Top-3 raw-policy actions at the state the AI just acted in."""
    game = env.game
    player = env.current_player
    static = _F.encode_static(game, player)
    history = _F.encode_history_steps(game)
    valid = env.get_valid_actions()
    probs, value = model.predict(static, history, valid)
    top = np.argsort(-probs)[:3]
    return {
        "top3": [{"label": action_obj(int(a))["label"], "p": round(float(probs[a]), 4)}
                 for a in top if probs[a] > 0],
        "value": [round(float(v), 3) for v in value],
        "chosen_p": round(float(probs[chosen]), 4),
    }


# ── state serialization ──────────────────────────────────────────────────────

def state_json(s, ai_moves=None):
    env = s["env"]
    g = env.game
    human = s["human_seat"]
    cur = env.current_player if not env.done else None

    last_play = None
    if g.control != 1 and g.goIndex > 1:
        hp = g.handsPlayed[g.goIndex - 1]
        last_play = {"player": int(hp.player),
                     "cards": [card_obj(c) for c in sorted(int(c) for c in hp.hand)]}

    legal = []
    if not env.done and cur == human:
        mask = env.get_valid_actions()
        legal = [action_obj(a) for a in np.flatnonzero(mask)]
        legal.sort(key=lambda x: (x["type"] == "pass", x["action"]))

    hist = []
    for h in g.actionHistory:
        if h["pass"]:
            hist.append({"player": int(h["player"]), "type": "pass",
                         "label": "PASS" + (" (auto)" if h.get("forced_skip") else ""),
                         "cards": []})
        else:
            cards = sorted(int(c) for c in h["hand"])
            nc = len(cards)
            t = "single" if nc == 1 else ("pair" if nc == 2 else "five")
            lbl = ("單張 " if nc == 1 else "對子 " if nc == 2 else f"{five_kind(cards)} ") + cards_str(cards)
            hist.append({"player": int(h["player"]), "type": t, "label": lbl,
                         "cards": [card_obj(c) for c in cards],
                         "control_break": bool(h.get("control_break"))})

    out = {
        "id": s["id"],
        "done": env.done,
        "human_seat": human,
        "current_player": cur,
        "control": int(g.control),
        "must_play_club3": bool(g.mustPlayClub3),
        "last_play": last_play,
        "counts": {p: int(g.currentHands[p].size) for p in range(1, 5)},
        "passed": {p: bool(g.passedThisRound[p]) for p in range(1, 5)},
        "hand": [card_obj(c) for c in sorted(int(c) for c in g.currentHands[human])],
        "legal": legal,
        "history": hist,
        "rewards": s["rewards"],
        "cfg": s["cfg"],
        "seed": s["seed"],
        "n_actions": len(s["actions"]),
    }
    if s["cfg"]["reveal"]:
        out["all_hands"] = {p: [card_obj(c) for c in sorted(int(c) for c in g.currentHands[p])]
                            for p in range(1, 5)}
    if ai_moves is not None:
        out["ai_moves"] = ai_moves
    return out


# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(os.path.join(WEB_DIR, "static"), "index.html")


@app.route("/static/<path:fn>")
def static_files(fn):
    return send_from_directory(os.path.join(WEB_DIR, "static"), fn)


@app.route("/api/checkpoints")
def api_checkpoints():
    out = []
    for name in list_checkpoints():
        meta = {"name": name}
        if name in _model_cache:
            meta["static_dim"] = _model_cache[name]["static_dim"]
            meta["has_value_net"] = _model_cache[name]["value_model"] is not None
        out.append(meta)
    return jsonify({"checkpoints": out, "default": DEFAULT_CFG["checkpoint"]})


@app.route("/api/game", methods=["POST"])
def api_new_game():
    body = request.get_json(force=True) or {}
    s = new_session(body)
    ai_moves = advance_ai(s)
    return jsonify(state_json(s, ai_moves=ai_moves))


@app.route("/api/game/<sid>", methods=["GET"])
def api_state(sid):
    return jsonify(state_json(get_session(sid)))


@app.route("/api/game/<sid>/play", methods=["POST"])
def api_play(sid):
    s = get_session(sid)
    body = request.get_json(force=True) or {}
    env = s["env"]
    if env.done:
        return jsonify({"error": "game over"}), 400
    if env.current_player != s["human_seat"]:
        return jsonify({"error": "not your turn"}), 400
    a = int(body["action"])
    mask = env.get_valid_actions()
    if a < 0 or a >= len(mask) or not mask[a]:
        return jsonify({"error": "illegal action"}), 400
    done = apply_action(s, a)
    ai_moves = [] if done else advance_ai(s)
    return jsonify(state_json(s, ai_moves=ai_moves))


@app.route("/api/game/<sid>/undo", methods=["POST"])
def api_undo(sid):
    """Rewind to just before the human's previous decision."""
    s = get_session(sid)
    acts = s["actions"]
    idx = [i for i, (p, _a, _t) in enumerate(acts) if p == s["human_seat"]]
    if not idx:
        return jsonify({"error": "nothing to undo"}), 400
    replay_to(s, idx[-1])
    return jsonify(state_json(s))


@app.route("/api/game/<sid>/analysis", methods=["GET"])
def api_analysis(sid):
    """Model's read of the CURRENT position from the human's perspective:
    top-k policy actions, 4-dim value, belief heatmap. Pure debug/hint —
    does not advance the game."""
    s = get_session(sid)
    env = s["env"]
    if env.done:
        return jsonify({"error": "game over"}), 400
    topk = int(request.args.get("topk", 8))
    player = s["human_seat"]
    with _model_lock:
        entry = load_checkpoint(s["cfg"]["checkpoint"])
        game = env.game
        static = _F.encode_static(game, player)
        history = _F.encode_history_steps(game)
        if env.current_player == player:
            valid = env.get_valid_actions()
        else:
            valid = None
        probs, value = entry["model"].predict(static, history, valid)
        bel = belief_probs(entry["model"], game, player)
    order = np.argsort(-probs)[:max(1, min(topk, 20))]
    others = [p for p in range(1, 5) if p != player]
    return jsonify({
        "for_player": player,
        "is_my_turn": env.current_player == player,
        "top_actions": [{**action_obj(int(a)), "p": round(float(probs[a]), 4)}
                        for a in order if probs[a] > 1e-4],
        "value": [round(float(v), 4) for v in value],
        "belief": {str(o): [round(float(x), 3) for x in bel[i]]
                   for i, o in enumerate(others)},
    })


@app.route("/api/game/<sid>/step", methods=["POST"])
def api_step(sid):
    """Advance exactly one AI move (manual stepping when it's not your turn —
    e.g. after undo into an AI turn, though advance_ai normally prevents that)."""
    s = get_session(sid)
    env = s["env"]
    if env.done or env.current_player == s["human_seat"]:
        return jsonify(state_json(s))
    with _model_lock:
        entry = load_checkpoint(s["cfg"]["checkpoint"])
        p = env.current_player
        a, dt = ai_action(entry, env, s["cfg"])
        apply_action(s, a, ai_time=dt)
    return jsonify(state_json(s, ai_moves=[{"player": p, **action_obj(a), "time": round(dt, 3)}]))


@app.route("/api/stats")
def api_stats():
    """Cumulative log stats (all finished web games)."""
    games = []
    if os.path.isfile(LOG_PATH):
        with open(LOG_PATH) as f:
            for line in f:
                try:
                    games.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    n = len(games)
    if n == 0:
        return jsonify({"games": 0})
    scores = [g["human_score"] for g in games]
    return jsonify({
        "games": n,
        "human_avg_score": round(float(np.mean(scores)), 3),
        "human_win_rate": round(float(np.mean([s > 0 for s in scores])), 3),
        "last10": scores[-10:],
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8288)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    print(f"checkpoints dir: {CKPT_DIR}")
    print(f"available: {', '.join(list_checkpoints()) or '(none!)'}")
    print(f"game log: {LOG_PATH}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
