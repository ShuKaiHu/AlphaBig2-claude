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
    # parsed_events gives us which seat is OUR agent (self_hand_snapshot)
    our_seat = None
    pe_path = os.path.join(os.path.dirname(path), "parsed_events.json")
    if os.path.exists(pe_path):
        try:
            pe = json.load(open(pe_path))
            snaps = [int(e["actor_index"]) for e in pe
                     if isinstance(e, dict) and e.get("event") == "self_hand_snapshot"
                     and str(e.get("actor_index", "")).isdigit()]
            if snaps:
                our_seat = max(set(snaps), key=snaps.count)
        except Exception:
            pass

    cur = None  # {"plays": [...], "played": {seat:set}, "settle": {seat:(score,ids)}}

    def finalize(g):
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
            g = finalize(cur)
            if g:
                yield g
            cur = {"plays": [], "played": {}, "settle": {}}
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
    g = finalize(cur)
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

    games = sorted(master.values(), key=lambda x: (x.get("run", ""), x.get("id", "")))
    with open(OUT, "w") as f:
        for g in games:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")

    n_dec = sum(len(g["plays"]) for g in games)
    print(f"runs scanned: {len(runs)}")
    print(f"previous master: {prev} games | new added: {added} | TOTAL: {len(games)} games")
    print(f"TOTAL decision points (plays+passes): {n_dec}  (avg {n_dec/max(len(games),1):.1f}/game)")
    print(f"master dataset (accumulating) -> {OUT}")


if __name__ == "__main__":
    main()
