"""Record store — the paper-literal historical state database (v4).

One record per real DECISION, with a SUCCESSOR POINTER to the same player's
next decision in the same game — the chains the prediction tree walks
(eq 1-3: states are rounds; eq 17-18: MP retrieval selects states into the
tree; the converging-reward recursion follows recorded transitions until
terminal).

Columns (parallel arrays, ~40 bytes/record → ~1.2 GB at 500K games):
  lead_id  : interned id of the combo leading the trick (None → 0 when leading)
  level    : Table-Card Level (0..49)
  own      : acting player's card count (1..13)
  opps_id  : interned id of the (o1,o2,o3) remaining-count tuple
  akey_id  : interned id of the action key taken (PASS included — an action
             like any other, per the paper's action set)
  won      : did this record's player win the game (uint8)
  score    : the player's final game score (float32; negative on a loss)
  npass    : how many of the NEXT THREE actors passed (0..3) — the R_c sample
             of eq 19-20 (K1+K2+K3), stored as a count, not a boolean
  succ     : record index of the same player's next decision, or -1 (terminal)

Retrieval (eq 17): a record is matched if it shares ≥1 of {lead, level, own}
with the query — or its FULL opponent-count triple (the conjunction clause).
The union is capped to the most recent `cap` records (documented tractability
assumption: the paper never states a retrieval bound).
"""
import pickle
from array import array
from collections import defaultdict


class _Intern:
    def __init__(self):
        self.map = {}
        self.items = []

    def id(self, key):
        i = self.map.get(key)
        if i is None:
            i = len(self.items)
            self.map[key] = i
            self.items.append(key)
        return i


class RecordStore:
    def __init__(self):
        self.lead = array("i")
        self.level = array("i")
        self.own = array("i")
        self.opps = array("i")
        self.akey = array("i")
        self.won = array("b")
        self.score = array("f")
        self.npass = array("b")
        self.succ = array("i")
        self._leads = _Intern()
        self._opps = _Intern()
        self._akeys = _Intern()
        # per-feature inverted indexes: value -> list of record indices
        self.ix_lead = defaultdict(lambda: array("i"))
        self.ix_level = defaultdict(lambda: array("i"))
        self.ix_own = defaultdict(lambda: array("i"))
        self.ix_opps = defaultdict(lambda: array("i"))
        self.n_games = 0

    # ── ingest ──────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self.akey)

    def akey_id(self, akey):
        return self._akeys.id(akey)

    def akey_of(self, aid):
        return self._akeys.items[aid]

    def add_record(self, feats, akey, won, score, npass, succ):
        lead, level, own, opps = feats
        i = len(self.akey)
        lid = self._leads.id(lead)
        oid = self._opps.id(opps)
        self.lead.append(lid); self.level.append(level)
        self.own.append(own); self.opps.append(oid)
        self.akey.append(self._akeys.id(akey))
        self.won.append(1 if won else 0)
        self.score.append(float(score))
        self.npass.append(int(npass))
        self.succ.append(int(succ))
        self.ix_lead[lid].append(i); self.ix_level[level].append(i)
        self.ix_own[own].append(i); self.ix_opps[oid].append(i)
        return i

    # ── retrieval (eq 17, feature-OR) ───────────────────────────────────────
    def retrieve(self, feats, cap=2000):
        lead, level, own, opps = feats
        lid = self._leads.map.get(lead)
        oid = self._opps.map.get(opps)
        pools = []
        if lid is not None:
            pools.append(self.ix_lead[lid])
        pools.append(self.ix_level.get(level, array("i")))
        pools.append(self.ix_own.get(own, array("i")))
        if oid is not None:
            pools.append(self.ix_opps[oid])
        per = max(1, cap // max(1, len(pools)))
        seen = set()
        for pool in pools:
            for i in pool[-per:]:
                seen.add(i)
        return sorted(seen)

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump({
                "cols": {k: getattr(self, k).tobytes() for k in
                         ("lead", "level", "own", "opps", "akey", "won",
                          "score", "npass", "succ")},
                "leads": self._leads.items,
                "opps": self._opps.items,
                "akeys": self._akeys.items,
                "n_games": self.n_games,
            }, f, protocol=4)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        st = cls()
        types = dict(lead="i", level="i", own="i", opps="i", akey="i",
                     won="b", score="f", npass="b", succ="i")
        for k, code in types.items():
            arr = array(code); arr.frombytes(blob["cols"][k])
            setattr(st, k, arr)
        for key in blob["leads"]:
            st._leads.id(key)
        for key in blob["opps"]:
            st._opps.id(key)
        for key in blob["akeys"]:
            st._akeys.id(key)
        for i in range(len(st.akey)):
            st.ix_lead[st.lead[i]].append(i)
            st.ix_level[st.level[i]].append(i)
            st.ix_own[st.own[i]].append(i)
            st.ix_opps[st.opps[i]].append(i)
        st.n_games = blob["n_games"]
        return st
