"""Learning-curve plot: pure function of Checkpoint rows and an output path."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt


def plot_learning_curve(checkpoints: Sequence[Any], path: Path) -> None:
    """Validation net pips against timesteps; eligible filled, ineligible
    hollow grey, the selected Checkpoint a star with a vertical guide."""
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        xs = [c.timestep for c in checkpoints]
        ys = [c.metrics.net_pips for c in checkpoints]
        ax.plot(xs, ys, color="tab:blue", linewidth=1.2, zorder=1)
        eligible = [c for c in checkpoints if c.position_changes > 0]
        ineligible = [c for c in checkpoints if c.position_changes == 0]
        ax.scatter(
            [c.timestep for c in eligible],
            [c.metrics.net_pips for c in eligible],
            color="tab:blue",
            label="eligible",
            zorder=2,
        )
        ax.scatter(
            [c.timestep for c in ineligible],
            [c.metrics.net_pips for c in ineligible],
            facecolors="none",
            edgecolors="grey",
            label="ineligible (no Position changes)",
            zorder=2,
        )
        for c in checkpoints:
            if c.selected:
                ax.axvline(c.timestep, color="tab:orange", linestyle="--", zorder=0)
                ax.scatter(
                    [c.timestep],
                    [c.metrics.net_pips],
                    marker="*",
                    s=220,
                    color="tab:orange",
                    label="selected",
                    zorder=3,
                )
        ax.set_xlabel("timesteps")
        ax.set_ylabel("validation net pips")
        ax.set_title("Learning curve")
        ax.legend()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=120, bbox_inches="tight")
    finally:
        plt.close(fig)
