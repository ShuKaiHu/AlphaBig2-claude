"""Parse 神來也 online games from Big2VisionAgent artifacts into a training dataset.

Big 2 is imperfect-information DURING play (you only see played cards), but at
game end 神來也 reveals every player's remaining hand via `showScore`. Played
cards (plsend) + revealed remaining (showScore) = each player's FULL 13-card hand
for the whole game. So we can turn imperfect-info online logs into FULL-INFO
training data: who held what, and how they played it.

Reads raw WS packets from each run's network_log.json:
  gsstart                              -> new game
  plsend <seat> <cardblob>             -> a play (cards public)
  plpass <seat>                        -> a pass
  showScore <seat> <score> [remblob..] -> settlement (remaining cards + score)

Output: one JSON line per VALID game (all four 13-card hands reconstructed):
  {run, game, our_seat, hands{seat:[card_ids]}, plays[{seat,action,cards}],
   scores{seat:int}, winner}
Card ids are our engine's 1..52 (id=(rank-1)*4+suit; rank 1=3 .. 13=2).
"""
import glob
import hashlib
import json
import os

ART = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude/artifacts"
OUT = os.path.join(os.path.dirname(__file__), "data", "online_games.jsonl")
VERSIONS = os.path.join(os.path.dirname(__file__), "data", "run_versions.json")


def _agent_label(run, overrides):
    """What our agent was in this run. Explicit override wins; else infer from the
    run.log ML note (mcts=old AlphaZero line, cardaware[+search]=PPO line)."""
    if run in overrides:
        return overrides[run]
    try:
        txt = open(os.path.join(ART, run, "autoplay_agent", "run.log")).read()
    except Exception:
        return "ours"
    if "ML: mcts" in txt:
        return "alpha"
    if "cardaware+search" in txt:
        return "cardaware+search"
    if "ML: cardaware" in txt:
        return "cardaware"
    return "ours"

_BV_SUIT = {"1": 4, "2": 3, "3": 2, "4": 1}
_BV_RANK = {"3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7,
            "T": 8, "J": 9, "Q": 10, "K": 11, "1": 12, "2": 13}


def cid(code):
    return (_BV_RANK[code[1]] - 1) * 4 + _BV_SUIT[code[0]]


def blob_ids(blob):
    return [cid(blob[i:i + 2]) for i in range(0, len(blob), 2)]


def parse_run(path):
    """Yield reconstructed games from one network_log.json."""
    try:
        nl = json.load(open(path))
    except Exception:
        return
    # Which seat is OUR agent, PER GAME. A run plays many games and re-seats at
    # every new table, so the seat changes within a run -- taking one mode over the
    # whole run (what this did before) was wrong for most games: measured against
    # reward_log's authoritative self_score, the old per-run seat was right only
    # 42.7% of the time. That fed bc_dataset(target="human"), which keeps the three
    # seats that are NOT ours, so a wrong seat silently trained the "human" imitation
    # target on our own agent's moves. self_hand_snapshot carries self_index, and
    # both logs share the WS `seq`, so we slice snapshots by each game's seq range.
    snaps = []          # [(seq, self_index)] ascending
    pe_path = os.path.join(os.path.dirname(path), "parsed_events.json")
    if os.path.exists(pe_path):
        try:
            pe = json.load(open(pe_path))
            for e in pe:
                if not isinstance(e, dict) or e.get("event") != "self_hand_snapshot":
                    continue
                idx = e.get("self_index", e.get("actor_index"))
                if str(idx).lstrip("-").isdigit() and str(e.get("seq", "")).isdigit():
                    snaps.append((int(e["seq"]), int(idx)))
            snaps.sort()
        except Exception:
            snaps = []

    def seat_for(lo, hi):
        """Our seat during the game spanning WS seq [lo, hi): the seat our own
        hand snapshots reported. None if this game had no snapshot to go on."""
        seen = [i for s, i in snaps if lo <= s < (hi if hi is not None else 1 << 62)]
        return max(set(seen), key=seen.count) if seen else None

    cur = None  # {"plays": [...], "played": {seat:set}, "settle": {seat:(score,ids)}}

    def finalize(g, hi=None):
        if not g or not g["settle"]:
            return None
        seats = [0, 1, 2, 3]
        hands = {}
        for s in seats:
            played = g["played"].get(s, [])
            rem = g["settle"].get(s, (None, []))[1]
            full = sorted(set(played) | set(rem))
            hands[s] = full
        # validate: 4 hands, 13 each, 52 distinct
        allc = [c for s in seats for c in hands[s]]
        if not all(len(hands[s]) == 13 for s in seats):
            return None
        if len(set(allc)) != 52:
            return None
        scores = {s: g["settle"][s][0] for s in seats if s in g["settle"]}
        if len(scores) != 4:
            return None
        our_seat = seat_for(g["seq0"], hi)
        if our_seat is None:
            return None          # no snapshot -> seat unknown; a guess would poison
                                 # the human-imitation target, so drop the game
        winner = max(scores, key=scores.get)
        return {"our_seat": our_seat, "hands": hands, "plays": g["plays"],
                "scores": scores, "winner": winner}

    for e in nl:
        if e.get("kind") != "ws_message":
            continue
        p = e.get("payload")
        if not isinstance(p, str):
            continue
        t = p.split()
        if not t:
            continue
        cmd = t[0]
        if cmd == "gsstart":
            seq = int(e["seq"]) if str(e.get("seq", "")).isdigit() else None
            g = finalize(cur, hi=seq)     # this gsstart bounds the previous game
            if g:
                yield g
            cur = {"plays": [], "played": {}, "settle": {}, "seq0": seq or 0}
        elif cmd == "plsend" and cur is not None and len(t) >= 3:
            try:
                seat = int(t[1]); ids = blob_ids(t[2])
            except Exception:
                continue
            cur["plays"].append({"seat": seat, "action": "play", "cards": ids})
            cur["played"].setdefault(seat, []).extend(ids)
        elif cmd == "plpass" and cur is not None and len(t) >= 2:
            try:
                seat = int(t[1])
            except Exception:
                continue
            cur["plays"].append({"seat": seat, "action": "pass", "cards": []})
        elif cmd == "showScore" and cur is not None and len(t) >= 3:
            try:
                seat = int(t[1]); score = int(t[2])
            except Exception:
                continue
            rem = blob_ids(t[3]) if len(t) > 3 and "," not in t[3] else []
            cur["settle"][seat] = (score, rem)
    g = finalize(cur, hi=None)
    if g:
        yield g


def game_id(g):
    """Stable content fingerprint so re-parsing never double-counts a game."""
    core = json.dumps({"hands": g["hands"], "plays": g["plays"]}, sort_keys=True)
    return hashlib.md5(core.encode()).hexdigest()[:16]


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # ACCUMULATE: load existing master so old games survive even if their raw
    # artifacts get pruned. Re-running only ADDS new games (dedup by content id).
    master = {}
    if os.path.exists(OUT):
        for line in open(OUT):
            line = line.strip()
            if line:
                r = json.loads(line)
                master[r.get("id") or game_id(r)] = r
    prev = len(master)

    runs = sorted(glob.glob(os.path.join(ART, "*", "autoplay_agent", "network_log.json")))
    added = 0
    for path in runs:
        run = path.split("/artifacts/")[1].split("/")[0]
        for g in parse_run(path):
            g["run"] = run
            gid = game_id(g)
            if gid not in master:
                g["id"] = gid
                master[gid] = g
                added += 1

    # (re)label seats for every game: our_seat -> agent version, others -> human.
    overrides = json.load(open(VERSIONS)) if os.path.exists(VERSIONS) else {}
    label_cache = {}
    for g in master.values():
        run = g.get("run", "")
        if run not in label_cache:
            label_cache[run] = _agent_label(run, overrides)
        seats = ["human", "human", "human", "human"]
        if g.get("our_seat") is not None:
            seats[g["our_seat"]] = label_cache[run]
        g["seats"] = seats

    games = sorted(master.values(), key=lambda x: (x.get("run", ""), x.get("id", "")))
    with open(OUT, "w") as f:
        for g in games:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    n_dec = sum(len(g["plays"]) for g in games)
    print(f"runs scanned: {len(runs)}")
    print(f"previous master: {prev} games | new added: {added} | TOTAL: {len(games)} games")
    print(f"TOTAL decision points (plays+passes): {n_dec}  (avg {n_dec/max(len(games),1):.1f}/game)")
    from collections import Counter
    by_agent = Counter(g["seats"][g["our_seat"]] for g in games if g.get("our_seat") is not None)
    human_dec = sum(1 for g in games for p in g["plays"] if g["seats"][p["seat"]] == "human")
    print("games by OUR-agent version:", dict(by_agent))
    print(f"human decision points (for imitation): {human_dec}")
    print(f"master dataset (accumulating) -> {OUT}")


if __name__ == "__main__":
    main()
