"""Record which agent version produced the artifact run(s) since a given time.

Used by play_and_parse.sh so future online games are tagged with the exact
version (V1/V2/V3/V4/...) rather than auto-guessed. Writes ppo/data/run_versions.json.

    python -m ppo.record_run <label> <since_epoch_seconds>
"""
import glob
import json
import os
import sys

ART = "/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude/artifacts"
VERS = os.path.join(os.path.dirname(__file__), "data", "run_versions.json")


def main():
    label = sys.argv[1]
    since = float(sys.argv[2])
    ov = json.load(open(VERS)) if os.path.exists(VERS) else {}
    n = 0
    for d in glob.glob(os.path.join(ART, "*", "autoplay_agent")):
        if os.path.getmtime(d) >= since - 5:
            run = d.split("/artifacts/")[1].split("/")[0]
            ov[run] = label
            n += 1
    os.makedirs(os.path.dirname(VERS), exist_ok=True)
    json.dump(ov, open(VERS, "w"), indent=2, ensure_ascii=False, sort_keys=True)
    print(f"recorded {n} run(s) as '{label}' -> {VERS}")


if __name__ == "__main__":
    main()
