"""Per-version online stats vs 神來也 humans, from run.log self_scores.

Aggregates EVERY scored round (run.log "Reward logged"), grouped by agent version
via run_versions.json. (More complete than the reconstructed dataset, which drops
games with capture gaps.) win = self_score > 0 (Big2: winner positive, losers negative).
"""
import glob
import json
import os
import re
import statistics as st

ART = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude/artifacts"
VERS = os.path.join(os.path.dirname(__file__), "data", "run_versions.json")
ORDER = ["V1", "V2", "V3", "V4"]


def main():
    overrides = json.load(open(VERS)) if os.path.exists(VERS) else {}
    scores = {}  # version -> [self_score,...]
    runs_by_ver = {}
    for path in sorted(glob.glob(os.path.join(ART, "*", "autoplay_agent", "run.log"))):
        run = path.split("/artifacts/")[1].split("/")[0]
        ver = overrides.get(run)
        if not ver:
            continue
        vals = [int(m) for m in re.findall(r"Reward logged: self_score=([+-]?\d+)", open(path).read())]
        if vals:
            scores.setdefault(ver, []).extend(vals)
            runs_by_ver.setdefault(ver, []).append((run, len(vals)))

    print(f"{'ver':4} {'n':>4} {'win%':>6} {'avg':>8} {'median':>7} {'std':>6} {'best':>6} {'worst':>7}")
    for ver in ORDER + [v for v in scores if v not in ORDER]:
        s = scores.get(ver)
        if not s:
            print(f"{ver:4}  (no data)")
            continue
        wins = sum(1 for x in s if x > 0)
        print(f"{ver:4} {len(s):>4} {100*wins/len(s):>5.1f}% {st.mean(s):>+8.2f} "
              f"{st.median(s):>+7.1f} {st.pstdev(s):>6.1f} {max(s):>+6d} {min(s):>+7d}")
    print()
    for ver in ORDER:
        if ver in runs_by_ver:
            print(f"  {ver} runs: " + ", ".join(f"{r}({n})" for r, n in runs_by_ver[ver]))


if __name__ == "__main__":
    main()
