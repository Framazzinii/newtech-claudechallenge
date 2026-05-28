#!/usr/bin/env python3
"""
Tournament runner — Build and Battle Your Trading Agent (Part 3)
PoliMi GSOM Quantitative Finance 2026

Usage (from repo root):
    python tools/run_tournament.py
    python tools/run_tournament.py --seed 42
    python tools/run_tournament.py --seed 42 --end-time 17:30:00 --out-dir results/

How it works:
  1. Builds a standard RMSC04 market with the MomentumAgents removed.
  2. Instantiates one agent per entry in AGENT_REGISTRY (see below) and
     attaches them to the config via config_add_agents(), which handles
     ID assignment and latency-model regeneration automatically.
  3. Runs the simulation.
  4. Reads the final mark-to-market value from every student agent and
     computes PnL = MTM - starting_cash.
  5. Prints a ranked table to stdout, writes:
       <out-dir>/tournament_results.csv
       <out-dir>/league_table.png

─────────────────────────────────────────────────────────────────────────────
INSTRUCTOR SETUP — edit AGENT_REGISTRY before the tournament session
─────────────────────────────────────────────────────────────────────────────
Each entry is a dict with three keys:
  "team"   - display name shown on the league table
  "cls"    - the agent class (must subclass TradingAgent)
  "kwargs" - extra constructor kwargs beyond the standard ones that the
             runner fills in automatically (id, symbol, starting_cash,
             name, type, random_state).  Leave as {} if none.

The three entries below use the sample student agents created in
student_work/.  Replace or extend this list with real student submissions.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── make repo root importable so student_work.* resolves ─────────────────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from abides_core import abides
from abides_markets.configs.rmsc04 import build_config
from abides_markets.utils import config_add_agents

# Student agent imports
from student_work.mean_reversion_agent import MeanReversionAgent
from student_work.trend_follower_agent import TrendFollowerAgent
from student_work.liquidity_provider_agent import LiquidityProviderAgent

# ─────────────────────────────────────────────────────────────────────────────
# AGENT_REGISTRY  ← instructors: add / replace entries here
# ─────────────────────────────────────────────────────────────────────────────
AGENT_REGISTRY: list[dict] = [
    {
        "team": "Group A - Mean Reversion",
        "cls": MeanReversionAgent,
        "kwargs": {},
    },
    {
        "team": "Group B - Trend Follower",
        "cls": TrendFollowerAgent,
        "kwargs": {},
    },
    {
        "team": "Group C - Liquidity Provider",
        "cls": LiquidityProviderAgent,
        "kwargs": {},
    },
    # ── add more teams below ─────────────────────────────────────────────────
    # {
    #     "team": "Group D – My Strategy",
    #     "cls": MyStrategyAgent,          # imported above
    #     "kwargs": {},
    # },
]
# ─────────────────────────────────────────────────────────────────────────────


# ── Simulation defaults ───────────────────────────────────────────────────────
DEFAULT_SEED       = 42
DEFAULT_END_TIME   = "17:30:00"
DEFAULT_OUT_DIR    = os.path.join(_REPO_ROOT, "log", "tournament")
STARTING_CASH      = 10_000_000   # cents — matches rmsc04 default
TICKER             = "ABM"


def build_tournament_config(seed: int, end_time: str) -> dict:
    """
    Build an RMSC04 config with MomentumAgents removed, then attach one
    student agent per registry entry.

    Agent IDs are assigned contiguously after the background agents so the
    kernel's latency model stays consistent.
    """
    # Step 1: standard RMSC04 background — no momentum agents
    cfg = build_config(
        seed=seed,
        end_time=end_time,
        num_momentum_agents=0,
        log_orders=False,
        exchange_log_orders=False,
        book_logging=False,
        stdout_log_level="WARNING",
        starting_cash=STARTING_CASH,
    )

    # Step 2: instantiate student agents
    base_id = len(cfg["agents"])   # first available ID
    student_agents = []

    # Fix the numpy random state so results are reproducible given the seed.
    # Each agent gets an independent RandomState derived from the master seed.
    rng = np.random.RandomState(seed + 9999)

    for i, entry in enumerate(AGENT_REGISTRY):
        agent_id   = base_id + i
        team_name  = entry["team"]
        cls        = entry["cls"]
        extra_kw   = entry.get("kwargs", {})

        agent = cls(
            id=agent_id,
            name=team_name,
            type=cls.__name__,
            symbol=TICKER,
            starting_cash=STARTING_CASH,
            random_state=np.random.RandomState(seed=rng.randint(0, 2**32)),
            **extra_kw,
        )
        student_agents.append(agent)

    # Step 3: attach to config (regenerates latency model)
    cfg = config_add_agents(cfg, student_agents)

    return cfg, student_agents


def collect_results(
    end_state: dict,
    student_agents: list,
) -> pd.DataFrame:
    """
    Extract final PnL for every student agent from the post-simulation
    end_state.  mark_to_market() uses the market close price that the
    exchange broadcasts to all agents at the end of the trading day.
    """
    # Index the live agent objects by ID for quick lookup
    live_agents = {a.id: a for a in end_state["agents"]}

    rows = []
    for reg_entry, template in zip(AGENT_REGISTRY, student_agents):
        agent = live_agents.get(template.id)
        if agent is None:
            continue

        # mark_to_market returns total value in cents (cash + marked shares)
        mtm   = agent.mark_to_market(agent.holdings)
        pnl   = mtm - agent.starting_cash
        shares = agent.holdings.get(TICKER, 0)

        rows.append(
            {
                "rank":         None,          # filled after sorting
                "team":         reg_entry["team"],
                "strategy":     reg_entry["cls"].__name__,
                "final_mtm":    mtm,
                "pnl_cents":    pnl,
                "pnl_dollars":  pnl / 100,
                "shares_held":  shares,
            }
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("pnl_cents", ascending=False)
        .reset_index(drop=True)
    )
    df["rank"] = df.index + 1
    return df


def print_league_table(df: pd.DataFrame) -> None:
    """Pretty-print the ranked PnL table to stdout."""
    print()
    print("=" * 62)
    print("  TOURNAMENT LEAGUE TABLE")
    print("=" * 62)
    print(f"  {'Rank':<5} {'Team':<32} {'PnL':>12}")
    print("-" * 62)
    for _, row in df.iterrows():
        sign   = "▲" if row["pnl_dollars"] >= 0 else "▼"
        colour = ""   # plain stdout; notebook will have colour
        print(
            f"  {int(row['rank']):<5} {row['team']:<32}"
            f"  {sign} ${row['pnl_dollars']:>10,.2f}"
        )
    print("=" * 62)
    print()


def plot_league_table(df: pd.DataFrame, out_path: str) -> None:
    """Save a horizontal bar chart ranked by PnL."""
    n   = len(df)
    # Reverse so best rank is at the top of the chart
    df_plot = df.iloc[::-1].reset_index(drop=True)

    colours = [
        "#2ecc71" if v >= 0 else "#e74c3c"
        for v in df_plot["pnl_dollars"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.55 + 1.5)))

    bars = ax.barh(
        df_plot["team"],
        df_plot["pnl_dollars"],
        color=colours,
        edgecolor="white",
        linewidth=0.6,
        height=0.6,
    )

    # Value labels on bars
    for bar, val in zip(bars, df_plot["pnl_dollars"]):
        label = f"${val:,.0f}"
        x_pos = bar.get_width()
        ha    = "left" if x_pos >= 0 else "right"
        offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0] or 1)
        ax.text(
            x_pos + (offset if x_pos >= 0 else -offset),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha=ha,
            fontsize=9,
        )

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Final PnL (USD)", fontsize=11)
    ax.set_title(
        "Tournament League Table — Final PnL per Team",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.tick_params(axis="y", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[tournament] League table saved → {out_path}")


def run(seed: int, end_time: str, out_dir: str) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)

    print(f"[tournament] Building config  seed={seed}  end_time={end_time}")
    cfg, student_agents = build_tournament_config(seed, end_time)
    n = len(AGENT_REGISTRY)
    print(f"[tournament] {n} team agent(s) registered: "
          + ", ".join(e["team"] for e in AGENT_REGISTRY))
    print(f"[tournament] Total agents in simulation: {len(cfg['agents'])}")

    print("[tournament] Running simulation…")
    end_state = abides.run(cfg)

    print("[tournament] Collecting results…")
    df = collect_results(end_state, student_agents)

    print_league_table(df)

    csv_path = os.path.join(out_dir, "tournament_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[tournament] Results saved      → {csv_path}")

    png_path = os.path.join(out_dir, "league_table.png")
    plot_league_table(df, png_path)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the ABIDES tournament and produce a PnL league table."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--end-time", type=str, default=DEFAULT_END_TIME,
        dest="end_time",
        help=f"Market close time HH:MM:SS (default: {DEFAULT_END_TIME})"
    )
    parser.add_argument(
        "--out-dir", type=str, default=DEFAULT_OUT_DIR,
        dest="out_dir",
        help=f"Output directory for CSV and PNG (default: {DEFAULT_OUT_DIR})"
    )
    args = parser.parse_args()
    run(seed=args.seed, end_time=args.end_time, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
