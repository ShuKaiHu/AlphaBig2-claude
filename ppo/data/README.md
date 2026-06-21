# Online human-game dataset (神來也大老二)

`online_games.jsonl` — one JSON object per fully-reconstructed real online game.
**Accumulating + durable**: `python -m ppo.parse_online_games` re-scans the
Big2VisionAgent artifacts and ADDS new games (dedup by content `id`), so games
survive even after their raw artifacts are pruned. This file is git-tracked.

Reconstruction is FULL-INFO: each player's played cards (public `plsend`) + their
revealed remaining cards at settlement (`showScore`) = their complete 13-card
hand. Every game is validated (4×13 cards, 52 distinct).

## Record schema
```json
{
  "id": "16-hex",            // stable content fingerprint (dedup key)
  "run": "20260620-211110",  // artifact session dir (timestamp)
  "our_seat": 2,             // seat (0-3) controlled by our agent that game
  "winner": 1,               // seat that emptied its hand
  "hands": {                 // FULL 13-card starting hand per seat (card ids 1-52)
     "0": [...13...], "1": [...], "2": [...], "3": [...] },
  "scores": {"0": -6, "1": 15, "2": -1, "3": -8},   // 神來也 multiplied score
  "plays": [                 // chronological action sequence
     {"seat": 0, "action": "play", "cards": [c1, c2, ...]},
     {"seat": 1, "action": "pass", "cards": []},
     ... ]
}
```

## Card id convention (matches our engine)
`id = (rank-1)*4 + suit`, rank 1..13 = 3,4,…,K,A,2 ; suit 1=♦ 2=♣ 3=♥ 4=♠.
So id 1 = 3♦ … id 52 = 2♠.

## Uses (full info enables many things)
- **Imitation / behavioral cloning** of any seat (`ppo/bc_dataset.py`, target=human/winner/all).
  IMPORTANT: training obs must be HALF-MASKED — only the acting player's own hand +
  public played cards + opponent card COUNTS. Full hands are used ONLY to
  reconstruct the acting player's hand + legal moves, never fed as opponent info.
- **Belief / opponent-hand modeling** — oracle target = opponents' actual hidden cards.
- **Offline RL** — (state, action, terminal score) tuples.
- **Opponent modeling / exploitation** — how specific seats play given their hands.
