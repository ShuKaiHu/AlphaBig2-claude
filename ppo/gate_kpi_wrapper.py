"""Gate-KPI wrapper (M3 Phase 0, Track C).

Launched DETACHED by ppo_trainer.py (--gate-every N): runs the FROZEN yardstick
eval_policy_gates.py on a temp checkpoint, then appends one JSON line
    {"update", "wall_time", "ckpt", "metrics", ...}
to ppo/data/rl_kpi_log.jsonl (append-only KPI card stream for the RL run).

This file never touches eval_policy_gates.py's metric definitions — it only
shells out to it and copies the card's "metrics" object verbatim.

Failure policy: NEVER raise. Any error is recorded as an {"error": ...} line in
the same log (so a dead gate run is visible) and the process exits 0 — the
training loop must not be able to crash or block because of a gate run.

Standalone usage (what the trainer launches):
    ./.venv/bin/python3 ppo/gate_kpi_wrapper.py --ckpt <tmp.pt> --update 30 \
        --games 2000 --workers 2 --log ppo/data/rl_kpi_log.jsonl
"""
import argparse
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))          # .../ppo
_ROOT = os.path.dirname(_HERE)                              # worktree root
GATES_SCRIPT = os.path.join(_ROOT, "eval_policy_gates.py")
DEFAULT_LOG = os.path.join(_HERE, "data", "rl_kpi_log.jsonl")


def _append_line(log_path, obj):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--update", type=int, required=True)
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--log", default=DEFAULT_LOG)
    ap.add_argument("--tag", default="", help="free-form run tag copied into the log line")
    args = ap.parse_args()
    # live online campaign + other jobs share this machine — hard cap
    args.workers = min(args.workers, 4)

    t0 = time.time()
    card_out = args.ckpt + ".gates.json"
    entry = {
        "update": args.update,
        "wall_time": None,           # filled at append time
        "ckpt": os.path.abspath(args.ckpt),
        "tag": args.tag,
    }
    try:
        # Preflight: verify the checkpoint is loadable BEFORE shelling out.
        # eval_policy_gates.py's mp pool respawns workers forever on an
        # unloadable ckpt (initializer FileNotFoundError loop), so without this
        # check a corrupt/missing ckpt burns CPU on the shared machine for the
        # full 3600s timeout before the {"error"} line lands. Fail fast instead.
        import torch  # local import: keeps wrapper importable without torch
        torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cmd = [sys.executable, GATES_SCRIPT,
               "--ckpt", args.ckpt,
               "--games", str(args.games),
               "--workers", str(args.workers),
               "--seed-base", str(args.seed_base),
               "--out", card_out]
        proc = subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True,
                              timeout=3600)
        if proc.returncode != 0:
            raise RuntimeError(
                f"eval_policy_gates exit {proc.returncode}: "
                f"{(proc.stderr or proc.stdout)[-2000:]}")
        with open(card_out) as f:
            card = json.load(f)
        entry["metrics"] = card["metrics"]
        entry["games"] = card["games"]
        entry["seed_base"] = card["seed_base"]
        entry["gate_seconds"] = round(time.time() - t0, 1)
    except Exception as e:                                   # noqa: BLE001
        entry["error"] = f"{type(e).__name__}: {e}"
        entry["gate_seconds"] = round(time.time() - t0, 1)
    try:
        entry["wall_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _append_line(args.log, entry)
    except Exception as e:                                   # noqa: BLE001
        print(f"[gate_kpi_wrapper] could not append to {args.log}: {e}",
              file=sys.stderr)
    sys.exit(0)


if __name__ == "__main__":
    main()
