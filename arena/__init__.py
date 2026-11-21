"""Prediction Arena: evolving trading bots on simulated prediction markets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"  # charts (the ones in the README are committed)
RUNS = ROOT / "runs"        # run outputs: histories, populations, trained agents (not in git)
FIGURES.mkdir(exist_ok=True)
RUNS.mkdir(exist_ok=True)
