"""
gui.py  —  Portfolio Backtester GUI
====================================
Run with:  python gui.py

Controls pass parameters to _gui_runner.py via a JSON argument.
The runner saves plots as PNGs and rebalancing detail as JSON files
inside a timestamped output directory.  After the run completes the
GUI loads those files into the Rebalancing History and Charts tabs.

Requires Pillow for the Charts tab:  pip install pillow
"""

import json
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk, scrolledtext

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RUNNER_PATH = os.path.join(SCRIPT_DIR, "_gui_runner.py")

# ── colour palette (Catppuccin Mocha) ────────────────────────────────────────
BG      = "#1e1e2e"
PANEL   = "#2a2a3e"
SURFACE = "#313244"
ACCENT  = "#89b4fa"   # blue
GREEN   = "#a6e3a1"
RED     = "#f38ba8"
FG      = "#cdd6f4"
FG2     = "#6c7086"
BORDER  = "#45475a"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: labelled slider
# ─────────────────────────────────────────────────────────────────────────────
class LabelledSlider(tk.Frame):
    def __init__(self, parent, label, var, from_, to, fmt, **kw):
        super().__init__(parent, bg=PANEL, **kw)
        self.var = var
        self.fmt = fmt

        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        tk.Label(hdr, text=label, font=("Segoe UI", 9, "bold"),
                 fg=FG, bg=PANEL).pack(side="left")
        self._val = tk.Label(hdr, text=fmt.format(var.get()),
                             font=("Segoe UI", 9), fg=ACCENT, bg=PANEL)
        self._val.pack(side="right")

        ttk.Scale(self, variable=var, from_=from_, to=to,
                  orient="horizontal",
                  command=lambda v: self._val.config(
                      text=fmt.format(float(v)))
                  ).pack(fill="x", pady=(2, 0))

        rng = tk.Frame(self, bg=PANEL)
        rng.pack(fill="x")
        tk.Label(rng, text=fmt.format(from_), font=("Segoe UI", 7),
                 fg=FG2, bg=PANEL).pack(side="left")
        tk.Label(rng, text=fmt.format(to), font=("Segoe UI", 7),
                 fg=FG2, bg=PANEL).pack(side="right")


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI class
# ─────────────────────────────────────────────────────────────────────────────
class BacktestGUI:
    def __init__(self, root: tk.Tk):
        self.root   = root
        self.root.title("Portfolio Backtester — GP2A")
        self.root.geometry("1300x800")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self._proc: subprocess.Popen | None = None
        self._running = False
        self._results_dir: str | None = None
        self._rebal_records: dict = {}
        self._chart_images: list = []   # keep PIL refs alive to prevent GC

        self._configure_styles()
        self._build_ui()

    # ── ttk style overrides ───────────────────────────────────────────────────
    def _configure_styles(self):
        st = ttk.Style()
        st.theme_use("clam")
        st.configure("TScale",       background=PANEL, troughcolor=SURFACE,
                     sliderthickness=14)
        st.configure("TRadiobutton", background=PANEL, foreground=FG,
                     font=("Segoe UI", 10))
        st.map("TRadiobutton",
               background=[("active", PANEL)],
               foreground=[("active", ACCENT)])
        st.configure("TCheckbutton", background=PANEL, foreground=FG,
                     font=("Segoe UI", 9))
        st.map("TCheckbutton",
               background=[("active", PANEL)])
        st.configure("Run.TButton",  background=ACCENT, foreground=BG,
                     font=("Segoe UI", 11, "bold"), padding=10)
        st.map("Run.TButton",
               background=[("active", "#74c7ec"), ("disabled", SURFACE)],
               foreground=[("disabled", FG2)])
        st.configure("Stop.TButton", background=RED, foreground=BG,
                     font=("Segoe UI", 10, "bold"), padding=8)
        st.map("Stop.TButton",
               background=[("active", "#eb6f92")])
        st.configure("TProgressbar", troughcolor=SURFACE, background=ACCENT,
                     thickness=6)
        st.configure("TNotebook",    background=BG, borderwidth=0)
        st.configure("TNotebook.Tab", background=PANEL, foreground=FG2,
                     padding=(12, 6), font=("Segoe UI", 9))
        st.map("TNotebook.Tab",
               background=[("selected", BG)],
               foreground=[("selected", ACCENT)])
        st.configure("Treeview",
                     background="#181825", foreground=FG,
                     rowheight=24, fieldbackground="#181825",
                     font=("Consolas", 9))
        st.configure("Treeview.Heading",
                     background=PANEL, foreground=ACCENT,
                     font=("Segoe UI", 9, "bold"))
        st.map("Treeview",
               background=[("selected", ACCENT)],
               foreground=[("selected", BG)])

    # ── top-level layout ──────────────────────────────────────────────────────
    def _build_ui(self):
        # Title strip
        title = tk.Frame(self.root, bg=BG, pady=12)
        title.pack(fill="x", padx=20)
        tk.Label(title, text="Portfolio Backtester",
                 font=("Segoe UI", 20, "bold"), fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(title, text="GP2A",
                 font=("Segoe UI", 9), fg=FG2, bg=BG).pack(side="left", padx=14)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True)

        # Left controls column
        left = tk.Frame(body, bg=PANEL, width=320)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)

        tk.Frame(body, bg=BORDER, width=1).pack(side="left", fill="y")

        # Right: notebook
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill="both", expand=True)

        self._tab_console = tk.Frame(self.notebook, bg=BG)
        self._tab_rebal   = tk.Frame(self.notebook, bg=BG)
        self._tab_charts  = tk.Frame(self.notebook, bg=BG)

        self.notebook.add(self._tab_console, text="  Console  ")
        self.notebook.add(self._tab_rebal,   text="  Rebalancing History  ")
        self.notebook.add(self._tab_charts,  text="  Charts  ")

        self._build_controls(left)
        self._build_console_tab(self._tab_console)
        self._build_rebal_tab(self._tab_rebal)
        self._build_charts_tab(self._tab_charts)

    # ── left controls panel ───────────────────────────────────────────────────
    def _build_controls(self, parent):
        canvas = tk.Canvas(parent, bg=PANEL, highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=PANEL)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        p = dict(padx=16, pady=5)

        # ── rebalancing period ────────────────────────────────────────────────
        self._heading(inner, "Rebalancing Period")
        self.mode_var = tk.StringVar(value="quarterly")
        for label, val in [
            ("Monthly",     "monthly"),
            ("Quarterly",   "quarterly"),
            ("Semi-Annual", "semi-annual"),
            ("Annual",      "annual"),
        ]:
            ttk.Radiobutton(inner, text=label,
                            variable=self.mode_var, value=val).pack(
                anchor="w", **p)

        self._divider(inner)

        # ── universe ──────────────────────────────────────────────────────────
        self._heading(inner, "Universe")
        self.parent_size_var = tk.DoubleVar(value=150)
        self.n_select_var    = tk.DoubleVar(value=50)
        LabelledSlider(inner, "Parent Universe Size",
                       self.parent_size_var, 50, 250,
                       "{:.0f} stocks").pack(fill="x", **p)
        LabelledSlider(inner, "Top N by Liquidity (investable)",
                       self.n_select_var, 10, 100,
                       "{:.0f} stocks").pack(fill="x", **p)

        self._divider(inner)

        # ── risk / return ─────────────────────────────────────────────────────
        self._heading(inner, "Risk & Return")
        self.cap_var        = tk.DoubleVar(value=10.0)
        self.target_vol_var = tk.DoubleVar(value=15.0)
        self.target_ret_var = tk.DoubleVar(value=20.0)
        self.use_ret_var    = tk.BooleanVar(value=False)

        LabelledSlider(inner, "Per-asset Cap",
                       self.cap_var, 2, 25,
                       "{:.1f} %").pack(fill="x", **p)
        LabelledSlider(inner, "Target Volatility",
                       self.target_vol_var, 5, 40,
                       "{:.1f} %/yr").pack(fill="x", **p)
        LabelledSlider(inner, "Target Return",
                       self.target_ret_var, 5, 50,
                       "{:.1f} %/yr").pack(fill="x", **p)

        ttk.Checkbutton(
            inner,
            text="Use target return to select \u03b3\n(instead of target vol)",
            variable=self.use_ret_var,
        ).pack(anchor="w", padx=16, pady=(2, 8))

        self._divider(inner)

        # ── run / stop buttons ────────────────────────────────────────────────
        btn_frame = tk.Frame(inner, bg=PANEL)
        btn_frame.pack(fill="x", padx=16, pady=10)

        self.run_btn = ttk.Button(btn_frame, text="\u25b6   Run Backtest",
                                  style="Run.TButton",
                                  command=self.start_run)
        self.run_btn.pack(fill="x", pady=(0, 6))

        self.stop_btn = ttk.Button(btn_frame, text="\u25a0   Stop",
                                   style="Stop.TButton",
                                   command=self.stop_run,
                                   state="disabled")
        self.stop_btn.pack(fill="x")

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(inner, textvariable=self.status_var,
                 font=("Segoe UI", 9), fg=FG2, bg=PANEL).pack(pady=(8, 2))
        self.progress = ttk.Progressbar(inner, mode="indeterminate")
        self.progress.pack(fill="x", padx=16, pady=(0, 16))

    # ── console tab ───────────────────────────────────────────────────────────
    def _build_console_tab(self, parent):
        hdr = tk.Frame(parent, bg=BG, pady=8)
        hdr.pack(fill="x", padx=16)
        tk.Label(hdr, text="Console Output",
                 font=("Segoe UI", 11, "bold"),
                 fg=FG, bg=BG).pack(side="left")
        tk.Button(hdr, text="Clear",
                  font=("Segoe UI", 8), bg=SURFACE, fg=FG2,
                  relief="flat", padx=10, pady=3,
                  command=self.clear_output).pack(side="right")

        self.output = scrolledtext.ScrolledText(
            parent,
            state="disabled",
            font=("Consolas", 9),
            bg="#0d0d1a", fg="#c8d3f5",
            insertbackground="white",
            wrap="word",
            relief="flat",
            padx=10, pady=10,
            selectbackground=ACCENT,
        )
        self.output.pack(fill="both", expand=True)

        self.output.tag_configure("ok",   foreground=GREEN)
        self.output.tag_configure("err",  foreground=RED)
        self.output.tag_configure("info", foreground=ACCENT)
        self.output.tag_configure("dim",  foreground=FG2)

    # ── rebalancing history tab ───────────────────────────────────────────────
    def _build_rebal_tab(self, parent):
        # Placeholder shown before results arrive
        self._rebal_placeholder = tk.Label(
            parent,
            text="Run a backtest to see rebalancing history.",
            font=("Segoe UI", 11), fg=FG2, bg=BG,
        )
        self._rebal_placeholder.pack(expand=True)

        # Main content frame (hidden until results load)
        self._rebal_content = tk.Frame(parent, bg=BG)

        # Strategy selector
        sel_frame = tk.Frame(self._rebal_content, bg=BG)
        sel_frame.pack(fill="x", padx=10, pady=(8, 2))
        tk.Label(sel_frame, text="Strategy:", font=("Segoe UI", 9, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        self._strategy_var = tk.StringVar(value="strategy_return")
        for label, val in [("Return-Seeking", "strategy_return"),
                           ("Risk-Targeted",  "strategy_risk")]:
            ttk.Radiobutton(sel_frame, text=label,
                            variable=self._strategy_var, value=val,
                            command=self._on_strategy_change).pack(
                side="left", padx=(8, 0))

        # Treeview for summary table
        tree_frame = tk.Frame(self._rebal_content, bg=BG)
        tree_frame.pack(fill="x", padx=10, pady=(4, 0))

        cols = ("date", "pre_value", "post_value", "turnover", "cost")
        self.rebal_tree = ttk.Treeview(tree_frame, columns=cols,
                                        show="headings", height=12,
                                        selectmode="browse")
        heads = {
            "date":       ("Date",         110, "center"),
            "pre_value":  ("Pre-Value ($)", 140, "e"),
            "post_value": ("Post-Value ($)",140, "e"),
            "turnover":   ("Turnover",      100, "e"),
            "cost":       ("Tx Cost ($)",   120, "e"),
        }
        for col, (label, width, anchor) in heads.items():
            self.rebal_tree.heading(col, text=label)
            self.rebal_tree.column(col, width=width, anchor=anchor, minwidth=60)

        tree_vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                                  command=self.rebal_tree.yview)
        self.rebal_tree.configure(yscrollcommand=tree_vsb.set)
        self.rebal_tree.pack(side="left", fill="x", expand=True)
        tree_vsb.pack(side="right", fill="y")
        self.rebal_tree.bind("<<TreeviewSelect>>", self._on_rebal_select)

        # Detail section heading
        tk.Label(self._rebal_content, text="Weight Changes",
                 font=("Segoe UI", 9, "bold"), fg=ACCENT, bg=BG,
                 anchor="w").pack(fill="x", padx=10, pady=(8, 2))

        # Weight changes text
        self.weight_text = scrolledtext.ScrolledText(
            self._rebal_content,
            state="disabled",
            font=("Consolas", 9),
            bg="#0d0d1a", fg="#c8d3f5",
            wrap="none",
            relief="flat",
            padx=10, pady=8,
            height=13,
        )
        self.weight_text.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self.weight_text.tag_configure("up",   foreground=GREEN)
        self.weight_text.tag_configure("down", foreground=RED)
        self.weight_text.tag_configure("hdr",  foreground=ACCENT,
                                       font=("Consolas", 9, "bold"))
        self.weight_text.tag_configure("dim",  foreground=FG2)

        # Store records per strategy
        self._all_rebal_records: dict[str, list[dict]] = {}

    # ── charts tab ────────────────────────────────────────────────────────────
    def _build_charts_tab(self, parent):
        msg = ("Run a backtest to see charts."
               if PIL_AVAILABLE else
               "Install Pillow (pip install pillow) then run a backtest to see charts.")
        self._charts_placeholder = tk.Label(
            parent, text=msg, font=("Segoe UI", 11), fg=FG2, bg=BG)
        self._charts_placeholder.pack(expand=True)

        # Scrollable canvas (hidden until results load)
        self._charts_outer = tk.Frame(parent, bg=BG)

        canvas = tk.Canvas(self._charts_outer, bg=BG, highlightthickness=0)
        vsb    = ttk.Scrollbar(self._charts_outer, orient="vertical",
                               command=canvas.yview)
        self._charts_inner = tk.Frame(canvas, bg=BG)

        self._charts_inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._charts_inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Mouse-wheel scrolling
        canvas.bind("<MouseWheel>",
                    lambda e: canvas.yview_scroll(-int(e.delta / 120), "units"))
        self._charts_canvas = canvas

    # ── layout helpers ────────────────────────────────────────────────────────
    def _heading(self, parent, text):
        tk.Label(parent, text=text.upper(),
                 font=("Segoe UI", 8, "bold"),
                 fg=ACCENT, bg=PANEL).pack(anchor="w", padx=16, pady=(14, 4))

    def _divider(self, parent):
        tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", padx=16, pady=4)

    # ── console helpers ───────────────────────────────────────────────────────
    def clear_output(self):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.configure(state="disabled")

    def _append(self, text, tag=None):
        self.output.configure(state="normal")
        self.output.insert("end", text, tag or "")
        self.output.see("end")
        self.output.configure(state="disabled")

    # ── run / stop ────────────────────────────────────────────────────────────
    def start_run(self):
        if self._running:
            return
        self._running = True
        self._results_dir = None
        self._all_rebal_records = {}
        self.run_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status_var.set("Running\u2026")
        self.progress.start(10)
        self.notebook.select(self._tab_console)

        config = {
            "mode":        self.mode_var.get(),
            "parent_size": int(self.parent_size_var.get()),
            "n_select":    int(self.n_select_var.get()),
            "cap":         round(float(self.cap_var.get()), 2),
            "target_vol":  round(float(self.target_vol_var.get()), 2),
            "target_ret":  round(float(self.target_ret_var.get()), 2),
            "use_ret":     bool(self.use_ret_var.get()),
        }

        self._append("\u2500" * 60 + "\n", "info")
        self._append("  Backtest configuration\n", "info")
        for k, v in config.items():
            self._append(f"    {k:<16} {v}\n", "dim")
        self._append("\u2500" * 60 + "\n\n", "info")

        threading.Thread(
            target=self._stream_subprocess,
            args=(config,),
            daemon=True,
        ).start()

    def stop_run(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._append("\n[Stopped by user]\n", "err")

    def _stream_subprocess(self, config):
        try:
            proc = subprocess.Popen(
                [sys.executable, RUNNER_PATH, json.dumps(config)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=SCRIPT_DIR,
            )
            self._proc = proc

            for line in proc.stdout:
                stripped = line.rstrip()

                # Sentinel: results directory is ready
                if stripped.startswith("GUI_RESULTS_DIR="):
                    results_dir = stripped.split("=", 1)[1]
                    self._results_dir = results_dir
                    self.root.after(300, self._load_results, results_dir)
                    self.root.after(0, self._append,
                                    f"[Results directory: {results_dir}]\n", "info")
                    continue

                # Sentinel: individual detail files (informational only)
                if stripped.startswith("GUI_DETAIL="):
                    continue

                low = line.lower()
                if any(w in low for w in ("error", "traceback", "exception")):
                    tag = "err"
                elif any(w in low for w in ("done", "complete", "finished")):
                    tag = "ok"
                else:
                    tag = None
                self.root.after(0, self._append, line, tag)

            proc.wait()
            rc  = proc.returncode
            msg = f"\n\u2713  Backtest complete (exit 0)\n" if rc == 0 \
                  else f"\n\u2717  Backtest failed (exit {rc})\n"
            self.root.after(0, self._append, msg, "ok" if rc == 0 else "err")

        except Exception as exc:
            self.root.after(0, self._append, f"\nGUI error: {exc}\n", "err")
        finally:
            self._proc = None
            self.root.after(0, self._run_done)

    def _run_done(self):
        self._running = False
        self.run_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress.stop()
        self.status_var.set("Ready")

    # ── results loading ───────────────────────────────────────────────────────
    def _load_results(self, results_dir: str):
        rd = Path(results_dir)

        # Load all rebalancing JSON files
        for jf in sorted(rd.rglob("rebalancing_*.json")):
            label = jf.stem.replace("rebalancing_", "")
            try:
                with open(jf, encoding="utf-8") as f:
                    records = json.load(f)
                self._all_rebal_records[label] = records
            except Exception as e:
                self.root.after(0, self._append,
                                f"[Warning: could not load {jf.name}: {e}]\n", "err")

        if self._all_rebal_records:
            self._show_rebal_content()

        # Load charts
        if PIL_AVAILABLE:
            png_files = sorted(rd.rglob("*.png"))
            if png_files:
                self._show_charts_content(png_files)

    def _show_rebal_content(self):
        if self._rebal_content.winfo_ismapped():
            pass
        else:
            self._rebal_placeholder.pack_forget()
            self._rebal_content.pack(fill="both", expand=True)

        # Populate for the currently selected strategy
        self._populate_rebal_tree(self._strategy_var.get())
        self.notebook.select(self._tab_rebal)

    def _populate_rebal_tree(self, strategy_key: str):
        # Clear existing rows
        for row in self.rebal_tree.get_children():
            self.rebal_tree.delete(row)
        self._rebal_records = {}

        records = self._all_rebal_records.get(strategy_key, [])
        for rec in records:
            date = rec["date"]
            self.rebal_tree.insert("", "end", iid=date, values=(
                date,
                f"{rec['pre_value']:>16,.0f}",
                f"{rec['post_value']:>16,.0f}",
                f"{rec['turnover']:.2%}",
                f"{rec['cost']:>14,.0f}",
            ))
            self._rebal_records[date] = rec

        # Clear weight detail
        self.weight_text.configure(state="normal")
        self.weight_text.delete("1.0", "end")
        self.weight_text.configure(state="disabled")

    def _on_strategy_change(self):
        self._populate_rebal_tree(self._strategy_var.get())

    def _on_rebal_select(self, event):
        sel = self.rebal_tree.selection()
        if not sel:
            return
        date = sel[0]
        rec  = self._rebal_records.get(date)
        if not rec:
            return

        old_w = rec.get("old_weights", {})
        new_w = rec.get("new_weights", {})
        all_tickers = sorted(set(old_w) | set(new_w))

        entered   = [(t, old_w.get(t, 0.0), new_w.get(t, 0.0))
                     for t in all_tickers
                     if new_w.get(t, 0.0) - old_w.get(t, 0.0) > 1e-4]
        exited    = [(t, old_w.get(t, 0.0), new_w.get(t, 0.0))
                     for t in all_tickers
                     if old_w.get(t, 0.0) - new_w.get(t, 0.0) > 1e-4]
        unchanged = [(t, new_w.get(t, 0.0))
                     for t in all_tickers
                     if abs(new_w.get(t, 0.0) - old_w.get(t, 0.0)) <= 1e-4
                     and new_w.get(t, 0.0) > 1e-8]

        entered.sort(key=lambda x: -(x[2] - x[1]))
        exited.sort(key=lambda x:  -(x[1] - x[2]))
        unchanged.sort(key=lambda x: -x[1])

        wt = self.weight_text
        wt.configure(state="normal")
        wt.delete("1.0", "end")

        wt.insert("end",
                  f"Rebalancing  {date}"
                  f"   Pre: ${rec['pre_value']:,.0f}"
                  f"   Post: ${rec['post_value']:,.0f}"
                  f"   TO: {rec['turnover']:.2%}"
                  f"   Cost: ${rec['cost']:,.0f}\n",
                  "hdr")
        wt.insert("end", "\u2500" * 72 + "\n\n", "dim")

        wt.insert("end", "INCREASED / ENTERED\n", "hdr")
        if entered:
            for t, old, new in entered:
                wt.insert("end",
                           f"  {t:>8s}  {old:6.2%}  \u2192  {new:6.2%}"
                           f"  (+{new - old:.2%})\n",
                           "up")
        else:
            wt.insert("end", "  (none)\n", "dim")

        wt.insert("end", "\nDECREASED / EXITED\n", "hdr")
        if exited:
            for t, old, new in exited:
                wt.insert("end",
                           f"  {t:>8s}  {old:6.2%}  \u2192  {new:6.2%}"
                           f"  ({new - old:.2%})\n",
                           "down")
        else:
            wt.insert("end", "  (none)\n", "dim")

        wt.insert("end", "\nUNCHANGED\n", "hdr")
        if unchanged:
            for t, w in unchanged:
                wt.insert("end", f"  {t:>8s}  {w:6.2%}\n", "dim")
        else:
            wt.insert("end", "  (none)\n", "dim")

        wt.configure(state="disabled")

    # ── charts ────────────────────────────────────────────────────────────────
    def _show_charts_content(self, png_files):
        if not self._charts_outer.winfo_ismapped():
            self._charts_placeholder.pack_forget()
            self._charts_outer.pack(fill="both", expand=True)

        # Clear old images
        for widget in self._charts_inner.winfo_children():
            widget.destroy()
        self._chart_images.clear()

        for png_path in png_files:
            try:
                img = Image.open(png_path)
                max_w = max(700, self._charts_outer.winfo_width() - 40)
                w, h = img.size
                if w > max_w:
                    ratio = max_w / w
                    img = img.resize((int(w * ratio), int(h * ratio)),
                                     Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._chart_images.append(photo)
                tk.Label(self._charts_inner, image=photo,
                         bg=BG).pack(pady=4)
                tk.Label(self._charts_inner, text=png_path.name,
                         font=("Segoe UI", 7), fg=FG2, bg=BG).pack()
            except Exception as e:
                tk.Label(self._charts_inner,
                         text=f"Error loading {png_path.name}: {e}",
                         fg=RED, bg=BG,
                         font=("Segoe UI", 9)).pack(pady=4)

        self._charts_canvas.yview_moveto(0)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    BacktestGUI(root)
    root.mainloop()
