"""
_gui_runner.py
--------------
Subprocess entry point called by gui.py.
Receives a JSON config string as argv[1], patches GP2A globals,
then runs the backtest.  Plots are saved as PNGs in a timestamped
output directory.  Results are signalled to the GUI via stdout sentinels.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

config = json.loads(sys.argv[1])

# ── set non-interactive backend BEFORE any matplotlib import ──────────────────
import matplotlib
matplotlib.use("Agg")

# ── import GP2A (creates its own OUT_DIR at import time) ──────────────────────
import GP2A as gp
from GP2A import Config

# ── replace CFG with GUI-selected settings ────────────────────────────────────
gp.CFG = Config(
    parent_size=int(config["parent_size"]),
    n_select=min(int(config["n_select"]), int(config["parent_size"])),
    cap=float(config["cap"]) / 100.0,
    target_vol_annual=float(config["target_vol"]) / 100.0,
)

# ── rebalance mode ─────────────────────────────────────────────────────────────
gp.RUN_MODES = config["mode"]

# ── display / output settings ─────────────────────────────────────────────────
gp.SHOW_PLOTS = False   # no interactive window in subprocess
gp.SAVE_PLOTS = True    # save PNGs to OUT_DIR

# ── run ───────────────────────────────────────────────────────────────────────
gp.main()
