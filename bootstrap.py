"""Path bootstrap — import this FIRST in every script here.

This workspace reuses the engine + the human-trained policy/obs code that live in
the AlphaBig2-ppo worktree (NOT copied). enumerateOptions loads `actionIndices.pkl`
via a CWD-relative path, so we chdir into AB2PPO for engine imports; our own data
and checkpoints are written via the ABSOLUTE paths exported here, so the chdir does
not affect where outputs land.
"""
import os
import sys

AB2PPO = os.environ.get("AB2PPO", "/Users/shukaihu/Code_Project_Local/AlphaBig2-ppo")
VALUE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(VALUE_DIR, "data")
CKPT_DIR = os.path.join(VALUE_DIR, "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

if AB2PPO not in sys.path:
    sys.path.insert(0, AB2PPO)
if VALUE_DIR not in sys.path:
    sys.path.insert(0, VALUE_DIR)  # keep our own modules importable after chdir
os.chdir(AB2PPO)  # enumerateOptions reads actionIndices.pkl relative to CWD
