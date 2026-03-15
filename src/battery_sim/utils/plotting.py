from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from battery_sim.utils.types import EvalResult


def plot_windows(
    windows: list[pd.DataFrame],
    figsize: tuple = (12, 5),
    price_col: str = "price",
    time_col: str = "timestamp",
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot sampled price windows with clear date ticks, each window as a separate segment."""
    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.tab20(np.linspace(0, 1, len(windows)))
    for i, w in enumerate(windows):
        ts = pd.to_datetime(w[time_col])
        ax.plot(ts, w[price_col].values, color=cmap[i], alpha=alpha, linewidth=0.8)
    ax.set_ylabel("Price ($/MWh)")
    ax.set_title(f"{len(windows)} Sampled Windows")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_eval(result: EvalResult, figsize: tuple = (14, 10)) -> plt.Figure:
    """Plot time series for a single evaluation: price, actions, SoC, cumulative revenue."""
    df = result.df
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    fig.suptitle(result.name, fontsize=14, fontweight="bold")

    # Price
    axes[0].plot(df["timestamp"], df["price"], linewidth=0.8)
    axes[0].set_ylabel("Price ($/MWh)")
    axes[0].grid(True, alpha=0.3)

    # Actions (colored by charge/discharge)
    colors = np.where(df["action"] > 0, "green", np.where(df["action"] < 0, "red", "gray"))
    axes[1].bar(df["timestamp"], df["action"], color=colors, width=0.03, alpha=0.7)
    axes[1].set_ylabel("Action (MW)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    # SoC
    axes[2].plot(df["timestamp"], df["soc"], linewidth=0.8, color="tab:orange")
    axes[2].set_ylabel("SoC")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(True, alpha=0.3)

    # Cumulative revenue
    axes[3].plot(df["timestamp"], df["cumulative_revenue"], linewidth=0.8, color="tab:green")
    axes[3].set_ylabel("Cumulative Revenue ($)")
    axes[3].grid(True, alpha=0.3)

    axes[3].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[3].xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def plot_daily_revenue(result: EvalResult, figsize: tuple = (10, 4)) -> plt.Figure:
    """Bar chart of daily revenue with mean line."""
    fig, ax = plt.subplots(figsize=figsize)
    daily = result.daily_revenue
    colors = np.where(daily.values >= 0, "tab:green", "tab:red")
    ax.bar(daily.index, daily.values, color=colors, alpha=0.7)
    ax.axhline(result.mean_daily_revenue, color="black", linestyle="--",
               label=f"Mean: ${result.mean_daily_revenue:,.2f}/day")
    ax.set_ylabel("Daily Revenue ($)")
    ax.set_title(f"{result.name} — Daily Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def compare_policies(
    results: list[EvalResult],
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Compare multiple policies: cumulative revenue, daily revenue distribution, actions."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Policy Comparison", fontsize=14, fontweight="bold")

    # 1. Cumulative revenue over time
    ax = axes[0, 0]
    for r in results:
        ax.plot(r.df["timestamp"], r.df["cumulative_revenue"], label=r.name, linewidth=0.8)
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.set_title("Cumulative Revenue")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # 2. Daily revenue box plot
    ax = axes[0, 1]
    daily_data = [r.daily_revenue.values for r in results]
    names = [r.name for r in results]
    bp = ax.boxplot(daily_data, labels=names, patch_artist=True)
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Daily Revenue ($)")
    ax.set_title("Daily Revenue Distribution")
    ax.grid(True, alpha=0.3)

    # 3. Summary bar chart
    ax = axes[1, 0]
    revenues = [r.cumulative_revenue for r in results]
    bars = ax.bar(names, revenues, color=colors, alpha=0.7)
    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"${rev:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Total Revenue ($)")
    ax.set_title("Total Revenue")
    ax.grid(True, alpha=0.3)

    # 4. SoC comparison
    ax = axes[1, 1]
    for r in results:
        ax.plot(r.df["timestamp"], r.df["soc"], label=r.name, linewidth=0.8, alpha=0.7)
    ax.set_ylabel("SoC")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Battery State of Charge")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
