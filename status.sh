#!/usr/bin/env bash
# Live health check for every line on STATUS.md — the board can go stale, processes don't lie.
# Run from anywhere: ./status.sh
R=/Users/shukaihu/Code_Project_Local/AlphaBig2-claude
V=/Users/shukaihu/Code_Project_Local/Big2VisionAgent-claude

echo "══════════ 線 1:AB2 線上測試 ══════════"
if pgrep -f "resume_ab2|run_awbc_ab2" >/dev/null; then echo "  process: RUNNING ✅"; else echo "  process: 停止 ⏸"; fi
for T in ab2_ctrl_p4500 ab2_awbc_T40last ab2_awbc_T40best; do
  N=$(grep -c "\"run_tag\": \"$T\"" "$V/artifacts/reward_log.jsonl" 2>/dev/null || echo 0)
  printf "  %-22s %s/300\n" "$T" "$N"
done

echo "══════════ 線 2:M3 Phase 1(RL)══════════"
if pgrep -f "ppo_trainer" >/dev/null; then echo "  訓練: RUNNING ✅"; else echo "  訓練: 未在跑"; fi
if [ -f "$R/ppo/data/rl_kpi_log.jsonl" ]; then
  echo "  最新 KPI 卡:"; tail -1 "$R/ppo/data/rl_kpi_log.jsonl" | cut -c1-160
else echo "  (尚無 KPI log)"; fi
ls -t "$R/ppo/checkpoints/" 2>/dev/null | grep -i "m3" | head -3 | sed 's/^/  ckpt: /'

echo "══════════ 其他背景 ══════════"
pgrep -f "eval_policy_gates" >/dev/null && echo "  gate 評估在跑" || true
pgrep -f "train_bc" >/dev/null && echo "  BC 訓練在跑" || true
D=$(wc -l < "$R/ppo/data/online_games.jsonl" 2>/dev/null | tr -d ' ')
echo "  收割 master: ${D:-?} 局(AB2 完賽後要補收)"

echo "══════════ git ══════════"
for REPO in "$R" "$V"; do
  cd "$REPO"
  printf "  %-24s %s  dirty:%s\n" "$(basename "$REPO")" "$(git status -sb | head -1)" "$(git status --porcelain | wc -l | tr -d ' ')"
done
