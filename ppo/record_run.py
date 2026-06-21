"""Record which agent version produced the artifact run(s) since a given time.

Used by play_and_parse.sh so future online games are tagged with the exact
version (V1/V2/V3/V4/...) rather than auto-guessed. Writes ppo/data/run_versions.json.

    python -m ppo.record_run <label> <since_epoch_seconds>
"""
import glob
import json
import os
import sys
import time

ART = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude/artifacts"
VERS = os.path.join(os.path.dirname(__file__), "data", "run_versions.json")


def _run_epoch(run):
    """Run dir name is its CREATION timestamp (YYYYMMDD-HHMMSS) — use that, not
    mtime: a long session's mtime is at its END, which leaked into the next
    session's window and mis-tagged it (the V2->V3 bug)."""
    try:
        return time.mktime(time.strptime(run, "%Y%m%d-%H%M%S"))
    except Exception:
        return 0.0


def main():
    label = sys.argv[1]
    since = float(sys.argv[2])
    ov = json.load(open(VERS)) if os.path.exists(VERS) else {}
    n = 0
    for d in glob.glob(os.path.join(ART, "*", "autoplay_agent")):
        run = d.split("/artifacts/")[1].split("/")[0]
        if _run_epoch(run) >= since - 60:   # tag only runs CREATED at/after launch
            ov[run] = label
            n += 1
    os.makedirs(os.path.dirname(VERS), exist_ok=True)
    json.dump(ov, open(VERS, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"recorded {n} run(s) as '{label}' -> {VERS}")


if __name__ == "__main__":
    main()
