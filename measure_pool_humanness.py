"""How human-like is each opponent-pool member? humanness = fraction of held-out
REAL human decisions (online_games.jsonl, target='human') where the model's own
greedy (argmax) action matches what the human actually played -- same definition
pool_registry.json already documents, just never applied to most members (only
PPO_V4_init/PPO_V6 have a recorded score; the newer V4PBV_Iter*/policy_mode1_iter*/
policy_all family/RL1/RL2 members never got measured).

Two record sets are built from the SAME corpus decisions (same underlying games,
same "human" target) so scores are directly comparable:
  - plain (obs, feats, pos)          -> for "cardaware" / "v6" arch members
  - history (obs, history, feats,pos) -> for "cardaware_history" arch members
Evaluating via direct forward() on precomputed batches (not env.greedy() per
decision) since bc_dataset.py's _Shim doesn't populate actionHistory -- this
avoids needing to extend that shim just for this diagnostic.

    ./.venv/bin/python measure_pool_humanness.py
"""
import numpy as np
import torch

from ppo.bc_dataset import build_records
from ppo.belief_history_dataset import DATA, build_belief_history_records
from ppo.network_cardaware import build_batch as build_batch_plain
from ppo.network_cardaware_history import build_batch_history
from ppo.pool_eval import POOL_MEMBERS, _build_net
import json


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print("building plain (non-history) human-decision records...")
    plain_recs = build_records(target="human")
    print("building history human-decision records...")
    games = [json.loads(l) for l in open(DATA)]
    hist_recs = build_belief_history_records(games, target="human")

    print(f"\n{'name':<20}{'arch':<18}{'n':>8}{'humanness':>12}")
    results = []
    for name, path, arch in POOL_MEMBERS:
        try:
            ck = torch.load(path, map_location=device, weights_only=False)
            net = _build_net(arch, device)
            net.load_state_dict(ck["model"])
            net.eval()
        except Exception as e:
            print(f"{name:<20} SKIP ({e})")
            continue

        with torch.no_grad():
            if arch == "cardaware_history":
                recs = hist_recs
                correct = 0
                for s in range(0, len(recs), 512):
                    mb = recs[s:s + 512]
                    b = build_batch_history(mb, device)
                    logits, _ = net.forward(b["obs"], b["history"], b["act_feats"], b["act_mask"])
                    correct += (logits.argmax(-1) == b["pos"]).sum().item()
                n = len(recs)
            elif arch == "v6":
                recs = plain_recs
                correct = 0
                for s in range(0, len(recs), 512):
                    mb = recs[s:s + 512]
                    b = build_batch_plain(mb, device)
                    logits, _, _ = net.forward(b["obs"], b["act_feats"], b["act_mask"])
                    correct += (logits.argmax(-1) == b["pos"]).sum().item()
                n = len(recs)
            else:  # cardaware
                recs = plain_recs
                correct = 0
                for s in range(0, len(recs), 512):
                    mb = recs[s:s + 512]
                    b = build_batch_plain(mb, device)
                    logits, _ = net.forward(b["obs"], b["act_feats"], b["act_mask"])
                    correct += (logits.argmax(-1) == b["pos"]).sum().item()
                n = len(recs)

        acc = correct / max(n, 1)
        results.append((name, arch, n, acc))
        print(f"{name:<20}{arch:<18}{n:>8}{acc*100:>11.1f}%")

    avg = np.mean([r[3] for r in results])
    print(f"\npool average humanness: {avg*100:.1f}%")
    print(f"members below 30% (near-random/bot-style): "
          f"{sum(1 for r in results if r[3] < 0.30)}/{len(results)}")
    print(f"members above 50% (clearly human-like): "
          f"{sum(1 for r in results if r[3] > 0.50)}/{len(results)}")


if __name__ == "__main__":
    main()
