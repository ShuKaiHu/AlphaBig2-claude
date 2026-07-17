#!/usr/bin/env python3
"""AlphaBig2 Training Dashboard — streamlit run dashboard.py"""
import re
import subprocess
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJ = Path(__file__).parent
LOGS = {
    "v6": PROJ / "logs" / "train_v6.log",
    "v7": PROJ / "logs" / "train_v7.log",
    "v8": PROJ / "logs" / "train_v8.log",
}
VERSION_COLORS = {"v6": "#4c9be8", "v7": "#e8884c", "v8": "#4ce89c"}

LOG_RE = re.compile(
    r"\[iter\s+(\d+)\]\s+(\S+)\s+reward=([+-]?\d+\.\d+)"
    r".*?p_loss=(\d+\.\d+)"
    r".*?v_loss=(\d+\.\d+)"
    r"(?:.*?b_loss=(\d+\.\d+))?"
    r"(?:.*?ent=(\d+\.\d+))?"
    r"(?:.*?T=(\d+\.\d+))?"
    r"(?:.*?buf=(\d+))?"
    r"(?:.*?\|\s*avg_score=([+-]?\d+\.\d+).*?win_rate=(\d+\.\d+))?"
    r"(?:.*?\[(\d+)s\])?"
)


def parse_log(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            rows.append({
                "iter":      int(m.group(1)),
                "mode":      m.group(2),
                "reward":    float(m.group(3)),
                "p_loss":    float(m.group(4)),
                "v_loss":    float(m.group(5)),
                "b_loss":    float(m.group(6)) if m.group(6) else None,
                "entropy":   float(m.group(7)) if m.group(7) else None,
                "temp":      float(m.group(8)) if m.group(8) else None,
                "buf":       int(m.group(9))   if m.group(9) else None,
                "avg_score": float(m.group(10)) if m.group(10) else None,
                "win_rate":  float(m.group(11)) if m.group(11) else None,
                "secs":      int(m.group(12))   if m.group(12) else None,
                "is_best":   "✓ best" in line,
            })
    return pd.DataFrame(rows)


def trainer_status() -> tuple[bool, str]:
    try:
        ps = subprocess.check_output(["ps", "aux"], text=True)
        procs = [l for l in ps.splitlines() if "engine.trainer" in l and "grep" not in l]
        if procs:
            pid = procs[0].split()[1]
            cpu = procs[0].split()[2]
            return True, f"PID {pid} · CPU {cpu}%"
        return False, "未執行"
    except Exception:
        return False, "未知"


def smooth(series: pd.Series, window: int = 10) -> pd.Series:
    return series.rolling(window, min_periods=1, center=True).mean()


# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AlphaBig2 Dashboard",
    page_icon="🃏",
    layout="wide",
)

# ─── Header ────────────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([8, 1])
with col_title:
    st.title("🃏 AlphaBig2 Training Dashboard")
with col_refresh:
    st.button("🔄 Refresh")

# ─── Load data ─────────────────────────────────────────────────────────────────
data: dict[str, pd.DataFrame] = {v: parse_log(p) for v, p in LOGS.items()}

# ─── Status bar ────────────────────────────────────────────────────────────────
running, status_msg = trainer_status()
status_cols = st.columns(4)

with status_cols[0]:
    if running:
        st.success(f"Trainer 運行中 · {status_msg}")
    else:
        st.warning(f"Trainer · {status_msg}")

for i, (ver, df) in enumerate(data.items(), start=1):
    with status_cols[i]:
        if df.empty:
            st.metric(ver, "無資料")
        else:
            last = df.iloc[-1]
            eval_rows = df.dropna(subset=["avg_score"])
            best_score = eval_rows["avg_score"].max() if not eval_rows.empty else None
            delta = f"best avg={best_score:+.2f}" if best_score is not None else ""
            st.metric(
                ver,
                f"iter {int(last['iter'])}",
                delta=delta,
                delta_color="normal",
            )

st.divider()

# ─── Smoothing slider ──────────────────────────────────────────────────────────
smooth_win = st.slider("平滑視窗 (iters)", 1, 30, 10)

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tab_overview, tab_v6, tab_v7, tab_v8 = st.tabs(["綜覽", "v6", "v7", "v8"])


def version_tab(ver: str, df: pd.DataFrame):
    if df.empty:
        st.info(f"找不到 {LOGS[ver]} 或尚無資料。")
        return

    # KPIs
    last = df.iloc[-1]
    eval_rows = df.dropna(subset=["avg_score"])

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("最新 iter", int(last["iter"]))
    k2.metric("最新 reward", f"{last['reward']:+.2f}")
    k3.metric("p_loss", f"{last['p_loss']:.4f}")
    k4.metric("v_loss", f"{last['v_loss']:.4f}")
    if not eval_rows.empty:
        best = eval_rows.loc[eval_rows["avg_score"].idxmax()]
        k5.metric("best avg_score", f"{best['avg_score']:+.2f}", f"iter {int(best['iter'])}")

    color = VERSION_COLORS[ver]

    # ── Reward chart ──────────────────────────────────────────────────────────
    fig_r = go.Figure()
    fig_r.add_trace(go.Scatter(
        x=df["iter"], y=df["reward"],
        mode="lines", line=dict(color=color, width=1, dash="dot"),
        name="reward (raw)", opacity=0.4,
    ))
    fig_r.add_trace(go.Scatter(
        x=df["iter"], y=smooth(df["reward"], smooth_win),
        mode="lines", line=dict(color=color, width=2),
        name=f"reward (smooth {smooth_win})",
    ))
    if not eval_rows.empty:
        fig_r.add_trace(go.Scatter(
            x=eval_rows["iter"], y=eval_rows["avg_score"],
            mode="markers+lines", marker=dict(size=8, symbol="diamond"),
            line=dict(color="#f5c542", width=1.5, dash="dash"),
            name="avg_score (eval)",
        ))
        best_row = eval_rows.loc[eval_rows["avg_score"].idxmax()]
        fig_r.add_vline(
            x=int(best_row["iter"]),
            line_dash="dot", line_color="gold",
            annotation_text=f"best {best_row['avg_score']:+.2f}",
            annotation_position="top right",
        )
    fig_r.update_layout(
        title="Reward & avg_score", xaxis_title="iter", yaxis_title="score",
        legend=dict(orientation="h"), height=350, margin=dict(t=40, b=0),
    )
    st.plotly_chart(fig_r, use_container_width=True)

    # ── Loss chart ────────────────────────────────────────────────────────────
    fig_l = make_subplots(
        rows=1, cols=2, subplot_titles=("Policy loss", "Value loss"),
        shared_xaxes=False,
    )
    fig_l.add_trace(go.Scatter(
        x=df["iter"], y=smooth(df["p_loss"], smooth_win),
        mode="lines", line=dict(color="#e86f6f", width=2), name="p_loss",
    ), row=1, col=1)
    fig_l.add_trace(go.Scatter(
        x=df["iter"], y=smooth(df["v_loss"], smooth_win),
        mode="lines", line=dict(color="#6fe8b5", width=2), name="v_loss",
    ), row=1, col=2)
    if df["b_loss"].notna().any():
        fig_l.add_trace(go.Scatter(
            x=df["iter"], y=smooth(df["b_loss"], smooth_win),
            mode="lines", line=dict(color="#c86fe8", width=2, dash="dot"), name="b_loss",
        ), row=1, col=1)
    fig_l.update_layout(height=300, margin=dict(t=30, b=0), showlegend=True)
    st.plotly_chart(fig_l, use_container_width=True)

    # ── Entropy & Temperature ─────────────────────────────────────────────────
    if df["entropy"].notna().any() or df["temp"].notna().any():
        fig_e = go.Figure()
        if df["entropy"].notna().any():
            fig_e.add_trace(go.Scatter(
                x=df["iter"], y=smooth(df["entropy"], smooth_win),
                mode="lines", line=dict(color="#e8c44c", width=2), name="entropy",
            ))
        if df["temp"].notna().any():
            fig_e.add_trace(go.Scatter(
                x=df["iter"], y=df["temp"],
                mode="lines", line=dict(color="#808080", width=1.5, dash="dot"), name="temp",
                yaxis="y2",
            ))
        fig_e.update_layout(
            title="Entropy & Temperature",
            yaxis=dict(title="entropy"),
            yaxis2=dict(title="temp", overlaying="y", side="right"),
            height=280, margin=dict(t=40, b=0), legend=dict(orientation="h"),
        )
        st.plotly_chart(fig_e, use_container_width=True)

    # ── Eval table ────────────────────────────────────────────────────────────
    if not eval_rows.empty:
        with st.expander("Eval 紀錄"):
            display = eval_rows[["iter", "avg_score", "win_rate", "reward", "is_best"]].copy()
            display["win_rate"] = (display["win_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(display, use_container_width=True, hide_index=True)


# ─── Overview tab ──────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Reward 趨勢比較（各版本）")
    fig_cmp = go.Figure()
    for ver, df in data.items():
        if df.empty:
            continue
        color = VERSION_COLORS[ver]
        fig_cmp.add_trace(go.Scatter(
            x=df["iter"], y=smooth(df["reward"], smooth_win),
            mode="lines", line=dict(color=color, width=2), name=f"{ver} reward",
        ))
        eval_rows = df.dropna(subset=["avg_score"])
        if not eval_rows.empty:
            fig_cmp.add_trace(go.Scatter(
                x=eval_rows["iter"], y=eval_rows["avg_score"],
                mode="markers", marker=dict(color=color, size=9, symbol="diamond"),
                name=f"{ver} avg_score",
            ))
    fig_cmp.update_layout(
        xaxis_title="iter", yaxis_title="score",
        legend=dict(orientation="h"), height=420, margin=dict(t=20, b=0),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # Best eval per version summary
    st.subheader("各版本最佳 Eval 摘要")
    summary_rows = []
    for ver, df in data.items():
        if df.empty:
            summary_rows.append({"version": ver, "total_iters": 0})
            continue
        eval_rows = df.dropna(subset=["avg_score"])
        row = {"version": ver, "total_iters": int(df["iter"].max())}
        if not eval_rows.empty:
            best = eval_rows.loc[eval_rows["avg_score"].idxmax()]
            row["best_avg_score"] = f"{best['avg_score']:+.2f}"
            row["best_win_rate"]  = f"{best['win_rate']:.1%}"
            row["best_at_iter"]   = int(best["iter"])
            row["eval_count"]     = len(eval_rows)
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

# ─── Per-version tabs ──────────────────────────────────────────────────────────
with tab_v6:
    version_tab("v6", data["v6"])
with tab_v7:
    version_tab("v7", data["v7"])
with tab_v8:
    version_tab("v8", data["v8"])
