"""Figures for the README and the paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import kelly  # noqa: E402
from .experiments import RECORD_CI, RECORD_P, RECORD_PAYOFF  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "font.size": 10, "legend.frameon": False,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pass_vs_win_rate(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.axvspan(*RECORD_CI, color=GRID, alpha=0.7, label="95% interval from 12 trades")
    ax.plot(table["win_rate"], table["pass"], color=BLUE, label="P(pass), exact")
    ax.axvline(RECORD_P, color=MUTED, linewidth=1, linestyle=":")
    ax.axvline(1 / (1 + RECORD_PAYOFF), color=ORANGE, linewidth=1, label="break-even win rate")
    ax.set_xlabel("True win rate")
    ax.set_ylabel("Probability of passing")
    ax.set_title("Challenge pass probability, payoff 2.28, risk 1% per trade", loc="left")
    ax.legend(loc="lower right")
    _save(fig, out / "pass_vs_win_rate.png")


def heatmap(table: pd.DataFrame, axis: str, out: Path) -> None:
    grid = table.pivot(index=axis, columns="win_rate", values="pass")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    mesh = ax.pcolormesh(grid.columns, grid.index, grid.values, cmap="Blues", vmin=0, vmax=1,
                         shading="nearest")
    contours = ax.contour(grid.columns, grid.index, grid.values, levels=[0.1, 0.5, 0.9],
                          colors=INK, linewidths=0.8)
    ax.clabel(contours, fmt=lambda v: f"{v:.0%}", fontsize=8)
    if axis == "payoff":
        b = np.linspace(grid.index.min(), grid.index.max(), 100)
        ax.plot(1 / (1 + b), b, color=ORANGE, linewidth=1.5, label="break-even")
        ax.legend(loc="upper right")
        ax.set_ylabel("Payoff ratio (average win / average loss)")
        title = "P(pass) by win rate and payoff, risk 1% per trade"
    else:
        ax.set_yscale("log")
        ax.set_yticks(grid.index, [f"{r:.2%}" for r in grid.index])
        ax.minorticks_off()
        ax.set_ylabel("Risk per trade (share of initial balance)")
        title = "P(pass) by win rate and risk per trade, payoff 2.28"
    ax.grid(False)
    ax.set_xlabel("Win rate")
    ax.set_title(title, loc="left")
    fig.colorbar(mesh, ax=ax, label="P(pass)")
    _save(fig, out / f"heatmap_{axis}.png")


def kelly_growth(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    f = np.linspace(0, 0.99, 500)
    for label, p, color in (("win rate 19% (CI low)", RECORD_CI[0], ORANGE),
                            ("win rate 41.7% (record)", RECORD_P, BLUE),
                            ("win rate 68% (CI high)", RECORD_CI[1], AQUA)):
        g = kelly.growth(f, p, RECORD_PAYOFF)
        ax.plot(f, g, color=color, label=label)
        f_star = kelly.kelly_fraction(p, RECORD_PAYOFF)
        if f_star > 0:
            ax.plot(f_star, kelly.growth(f_star, p, RECORD_PAYOFF), "o", color=color, markersize=6)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylim(-0.3, 0.4)
    ax.set_xlabel("Fraction of current balance risked per trade, f")
    ax.set_ylabel("Expected log growth per trade, g(f)")
    ax.set_title("Kelly growth curves, payoff 2.28 (dots mark f*)", loc="left")
    ax.legend(loc="lower left")
    _save(fig, out / "kelly_growth.png")


def estimation_error(table: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["n_trades"], table["best_multiplier"], color=BLUE, marker="o", markersize=5)
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Trades used to estimate the win rate")
    ax.set_ylabel("Best multiple of the estimated Kelly stake")
    ax.set_title("Bet a fraction of Kelly when the win rate is estimated", loc="left")
    _save(fig, out / "kelly_estimation_error.png")
