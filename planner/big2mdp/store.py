"""Statistics store — the replication's version of Big2MDP's N(s,a,s') tables.

One record per real decision in a finished self-play game:
  features (lead_key, level, own, opps) + action_key + outcome
  (won?, final_score) + next3pass (did the next three actors all pass? →
  the R_c "kept the free-playing right" sample, eq 19-20).

Retrieval follows the paper's MP feature-OR matching (eq 17): a historical
record is relevant if it shares AT LEAST ONE of: lead_key, level, own count —
or ALL THREE opponent counts (the eq 17 conjunction clause).

ENGINEERING DEVIATION (documented): instead of storing 20M raw records and
unioning index sets per query, we keep per-feature aggregate tables
dict[(feature_value, action_key)] → [n, wins, win_score_sum, loss_score_sum,
next3pass_sum] and SUM the four feature tables' rows at query time. A record
matching k>1 features is counted k times — a weighted-union approximation of
the paper's set union (weights = how many features matched, which if anything
favors more-similar records). Memory stays tiny and queries are O(1).
"""
import pickle
from collections import defaultdict


def _zeros():
    return [0, 0, 0.0, 0.0, 0]


class StatsStore:
    FEATS = ("lead", "level", "own", "opps")

    def __init__(self):
        self.tables = {f: defaultdict(_zeros) for f in self.FEATS}
        self.n_games = 0
        self.n_records = 0

    # ── ingest ──────────────────────────────────────────────────────────────
    def add_record(self, feats, akey, won, score, next3pass):
        lead, level, own, opps = feats
        row_keys = (("lead", (lead, akey)), ("level", (level, akey)),
                    ("own", (own, akey)), ("opps", (opps, akey)))
        for tname, key in row_keys:
            row = self.tables[tname][key]
            row[0] += 1
            if won:
                row[1] += 1
                row[2] += score
            else:
                row[3] += score          # score is negative on a loss
            row[4] += int(next3pass)
        self.n_records += 1

    # ── query ───────────────────────────────────────────────────────────────
    def stats(self, feats, akey):
        """→ dict(n, p_win, r_win, p_lose, r_lose, r_c) for one candidate
        action key under MP feature-OR matching; n==0 means no data."""
        lead, level, own, opps = feats
        n = wins = 0
        win_sum = loss_sum = 0.0
        pass3 = 0
        for tname, fval in (("lead", lead), ("level", level),
                            ("own", own), ("opps", opps)):
            row = self.tables[tname].get((fval, akey))
            if row:
                n += row[0]; wins += row[1]
                win_sum += row[2]; loss_sum += row[3]
                pass3 += row[4]
        if n == 0:
            return dict(n=0, p_win=0.0, r_win=0.0, p_lose=0.0, r_lose=0.0, r_c=0.0)
        losses = n - wins
        return dict(
            n=n,
            p_win=wins / n,
            r_win=(win_sum / wins) if wins else 0.0,
            p_lose=losses / n,
            r_lose=(loss_sum / losses) if losses else 0.0,
            r_c=pass3 / n,
        )

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({"tables": {k: dict(v) for k, v in self.tables.items()},
                         "n_games": self.n_games,
                         "n_records": self.n_records}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        st = cls()
        for k, d in blob["tables"].items():
            st.tables[k] = defaultdict(_zeros, d)
        st.n_games = blob["n_games"]
        st.n_records = blob["n_records"]
        return st
