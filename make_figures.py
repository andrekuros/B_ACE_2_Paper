#!/usr/bin/env python3
"""Generate paper figures: WEZ, FSM, MARL, and tactical visualizer views."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

GDM2NM = 100.0 / 1852.0
RADAR_NM = 50.0
RADAR_HFOV = 60.0


def gdm_to_plot(pos) -> tuple[float, float]:
    """X east (NM), north = -Z (NM) so heading 0 points up."""
    return float(pos[0]) * GDM2NM, -float(pos[2]) * GDM2NM


def binom_ci(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def load_json(name: str) -> dict:
    path = ROOT / "data" / name
    if path.is_file():
        return json.loads(path.read_text())
    return {}


def run_wez_paper() -> dict:
    cached = load_json("paper_results.json")
    if cached.get("wez_cells"):
        return {
            "cells": cached["wez_cells"],
            "fits": cached.get("wez_fits", []),
            "traces": cached.get("wez_traces", []),
            "summary": cached.get("wez_summary"),
            "aspect_summaries": cached.get("wez_aspect_summaries"),
        }
    from bace.experiment import run_wez

    return run_wez(profile="paper", max_parallel=8, out_dir=ROOT / "data")


def run_fsm_paper() -> dict:
    cached = load_json("paper_results.json")
    if cached.get("fsm_history"):
        return {
            "history": cached["fsm_history"],
            "elites": cached.get("fsm_elites", []),
            "summary": cached.get("fsm_summary"),
            "params": cached.get("fsm_params"),
            "last_generation": [],
        }
    from bace.experiment import run_fsm

    return run_fsm(profile="paper", max_parallel=8, out_dir=ROOT / "data")


def load_marl_paper() -> dict:
    ppo = load_json("marl_ppo_1v1.json")
    core = load_json("marl_core.json")
    jobs = list(ppo.get("jobs") or []) + list(core.get("jobs") or [])
    cached = load_json("paper_results.json")
    if not jobs:
        jobs = list((cached.get("marl_core") or {}).get("jobs") or [])
        core = {**core, **(cached.get("marl_core") or {})}
    return {
        "recipe": "marl_core",
        "jobs": jobs,
        "ppo": ppo,
        "core": core,
        "n_jobs": len(jobs),
        "duck_2v2_learned": core.get("duck_2v2_learned"),
        "note": core.get("note") or ppo.get("note") or (cached.get("marl_core") or {}).get("note"),
    }


def plot_wez(report: dict) -> None:
    import matplotlib.pyplot as plt

    cells = report["cells"]
    alts = sorted({c["altitude_ft"] for c in cells})
    aspects = ["head", "beam", "tail"]
    colors = {"head": "#1f77b4", "beam": "#ff7f0e", "tail": "#2ca02c"}
    fig, axes = plt.subplots(1, len(alts), figsize=(10.2, 3.4), sharey=True)
    if len(alts) == 1:
        axes = [axes]
    fits = report.get("fits", [])
    for ax, alt in zip(axes, alts):
        for asp in aspects:
            rows = sorted(
                [c for c in cells if c["aspect"] == asp and abs(c["altitude_ft"] - alt) < 1],
                key=lambda c: c["range_nm"],
            )
            if not rows:
                continue
            xs = [c["range_nm"] for c in rows]
            ys = [c["hit_rate"] for c in rows]
            err = [binom_ci(c["hit_rate"], c["n"]) for c in rows]
            ax.errorbar(xs, ys, yerr=err, marker="o", lw=1.4, capsize=2, color=colors[asp], label=asp)
            rmax = rows[0]["analytic_rmax_nm"]
            ax.axvline(rmax, color=colors[asp], ls="--", lw=0.9, alpha=0.7)
            fit = next(
                (f for f in fits if f["aspect"] == asp and abs(f["altitude_ft"] - alt) < 1),
                None,
            )
            if fit and fit.get("poly_coef"):
                coef = fit["poly_coef"]
                grid = np.linspace(min(xs), max(xs), 80)
                poly = np.poly1d(coef)
                ax.plot(grid, np.clip(poly(grid), 0, 1), color=colors[asp], ls=":", lw=1.3, alpha=0.9)
        ax.set_title(f"{int(alt/1000)} kft")
        ax.set_xlabel("Launch range (NM)")
        ax.set_xlim(6, 42)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Hit rate")
    axes[-1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "wez_hitrate.pdf")
    fig.savefig(FIG / "wez_hitrate.png", dpi=160)
    plt.close(fig)


def plot_wez_fit(report: dict) -> None:
    import matplotlib.pyplot as plt

    fits = [f for f in report.get("fits", []) if abs(f.get("altitude_ft", 0) - 25000) < 1]
    if not fits:
        return
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    labels = [f["aspect"] for f in fits]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, [f["empirical_rmax_nm"] for f in fits], w, label=r"empirical $R_{\max}$", color="#1f77b4")
    ax.bar(x + w / 2, [f["analytic_rmax_nm"] for f in fits], w, label=r"analytic $R_{\max}$", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Range (NM)")
    ax.set_title(r"Fitted vs analytic $R_{\max}$ at 25 kft")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "wez_fit.pdf")
    fig.savefig(FIG / "wez_fit.png", dpi=160)
    plt.close(fig)


def plot_wez_traces(report: dict) -> None:
    import matplotlib.pyplot as plt

    traces = report.get("traces", [])
    if not traces:
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    for ax, rng in zip(axes, (16.0, 40.0)):
        rows = [t for t in traces if abs(t["range_nm"] - rng) < 0.4]
        hits = [t for t in rows if t.get("hit")]
        misses = [t for t in rows if not t.get("hit")]
        ax.hist(
            [t["tof"] for t in hits],
            bins=12,
            alpha=0.7,
            label=f"hit n={len(hits)}",
            color="#2ca02c",
        )
        ax.hist(
            [t["tof"] for t in misses],
            bins=12,
            alpha=0.55,
            label=f"miss n={len(misses)}",
            color="#d62728",
        )
        causes = {}
        for t in misses:
            causes[t.get("miss_cause") or "unknown"] = causes.get(t.get("miss_cause") or "unknown", 0) + 1
        ax.set_title(f"{int(rng)} NM  " + ", ".join(f"{k}:{v}" for k, v in causes.items()))
        ax.set_xlabel("Time of flight (s)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(FIG / "wez_traces.pdf")
    fig.savefig(FIG / "wez_traces.png", dpi=160)
    plt.close(fig)


def plot_fsm(report: dict) -> None:
    import matplotlib.pyplot as plt

    hist = report.get("history", [])
    last = report.get("last_generation", [])
    elites = {e["label"]: e for e in report.get("elites", [])}
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    if hist:
        axes[0].plot([h["generation"] for h in hist], [h["best_fitness"] for h in hist], "k-o", ms=4)
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel(r"Best fitness $0.5(K-D)+M-0.1S$")
    axes[0].grid(True, alpha=0.3)
    n = (report.get("params") or {}).get("num_agents", "?")
    axes[0].set_title(f"FSM search ({n}v{n})")
    if last:
        axes[1].scatter(
            [r["mean_deaths"] for r in last],
            [r["mean_kills"] for r in last],
            c="#8aa0b5",
            s=28,
            label="population",
        )
        markers = {"aggressive": ("^", "#d62728"), "balanced": ("o", "#1f77b4"), "cautious": ("s", "#2ca02c")}
        for lab, (mk, col) in markers.items():
            e = elites.get(lab)
            if not e:
                continue
            axes[1].scatter(
                [e["mean_deaths"]],
                [e["mean_kills"]],
                marker=mk,
                c=col,
                s=70,
                zorder=3,
                label=lab,
            )
    axes[1].set_xlabel("Mean deaths / episode")
    axes[1].set_ylabel("Mean kills / episode")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Last generation")
    fig.tight_layout()
    fig.savefig(FIG / "fsm_search.pdf")
    fig.savefig(FIG / "fsm_search.png", dpi=160)
    plt.close(fig)


def plot_fsm_occupancy(report: dict) -> None:
    import matplotlib.pyplot as plt

    elites = report.get("elites", [])
    if not elites:
        return
    labels = [e.get("label", "?") for e in elites]
    keys = [("fsm_search", "search"), ("fsm_engage", "engage"), ("fsm_support", "support"), ("fsm_evade", "evade")]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    colors = ["#8aa0b5", "#1f77b4", "#ffd166", "#d62728"]
    last = report.get("last_generation", [])
    src_rows = []
    for e in elites:
        match = next((r for r in last if r.get("genome") == e.get("genome")), None)
        src_rows.append(match or e.get("eval_4v4") or e)
    for (key, name), col in zip(keys, colors):
        vals = []
        for row in src_rows:
            vals.append(float(row.get(key, 0.0)))
        ax.bar(x, vals, bottom=bottom, color=col, label=name)
        bottom = bottom + np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("State occupancy")
    ax.set_title("FSM state mix (search generation)")
    ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fsm_occupancy.pdf")
    fig.savefig(FIG / "fsm_occupancy.png", dpi=160)
    plt.close(fig)


def plot_marl(report: dict) -> None:
    import matplotlib.pyplot as plt

    jobs = report.get("jobs") or report.get("core", {}).get("jobs") or []
    if not jobs and report.get("returns"):
        jobs = [report]
    groups = []
    for j in jobs:
        key = (int(j.get("agents", 1)), str(j.get("opponent", "duck")), str(j.get("algo", "ppo")))
        if key not in groups:
            groups.append(key)
    if not groups:
        return
    n = len(groups)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.2 * rows), squeeze=False)
    for i, key in enumerate(groups):
        ax = axes[i // cols][i % cols]
        sz, opp, algo = key
        series = [
            jj
            for jj in jobs
            if int(jj.get("agents", 0)) == sz
            and str(jj.get("opponent")) == opp
            and str(jj.get("algo")) == algo
        ]
        kill_curves = [np.asarray(s.get("kills_curve") or [c.get("kills", 0) for c in s.get("curve") or []], dtype=float) for s in series]
        kill_curves = [c for c in kill_curves if len(c) >= 2]
        if kill_curves:
            mlen = min(len(c) for c in kill_curves)
            stacked = np.stack([c[:mlen] for c in kill_curves])
            mu, sd = stacked.mean(0), stacked.std(0)
            xs = np.arange(1, mlen + 1)
            ax.plot(xs, mu, color="#1f77b4", lw=1.6, label="eval kills")
            ax.fill_between(xs, mu - sd, mu + sd, color="#1f77b4", alpha=0.15)
        ax.set_title(f"{algo} {sz}v{sz} vs {opp}")
        ax.set_ylabel("Eval kills / episode")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    for j in range(i + 1, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "marl_curve.pdf")
    fig.savefig(FIG / "marl_curve.png", dpi=160)
    plt.close(fig)


def plot_marl_eval(report: dict) -> None:
    import matplotlib.pyplot as plt

    jobs = report.get("jobs") or report.get("core", {}).get("jobs") or []
    if not jobs:
        return
    labels = []
    kills, rnd, fire, ret = [], [], [], []
    for j in jobs:
        sz = int(j.get("agents", 1))
        opp = str(j.get("opponent", "duck"))
        algo = str(j.get("algo", "ppo"))
        seed = j.get("seed")
        ev = j.get("eval") or {}
        labels.append(f"{algo}\n{sz}v{sz}/{opp}\ns{seed}")
        kills.append(float(ev.get("kills", 0)))
        rnd.append(float((j.get("eval_random") or {}).get("kills", 0)))
        fire.append(float((j.get("eval_fire_once") or {}).get("kills", 0)))
        ret.append(float(ev.get("return", 0)))
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7.0, 0.7 * len(labels)), 3.8))
    ax.bar(x - w, kills, w, label="policy kills", color="#1f77b4")
    ax.bar(x, rnd, w, label="random kills", color="#ff7f0e")
    ax.bar(x + w, fire, w, label="fire_once kills", color="#2ca02c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Kills / episode (50 eval)")
    ax.set_title("Frozen eval vs random and FireOnce")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "marl_eval.pdf")
    fig.savefig(FIG / "marl_eval.png", dpi=160)
    plt.close(fig)


def plot_bench(report: dict) -> None:
    import matplotlib.pyplot as plt

    scaling = report.get("scaling") or []
    if not scaling:
        return
    ns = [r["n_envs"] for r in scaling]
    hz = [r["decision_hz"] for r in scaling]
    rt = [r["realtime_factor"] for r in scaling]
    eff = [r.get("efficiency", r["realtime_factor"] / max(r["n_envs"], 1)) for r in scaling]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
    axes[0].plot(ns, hz, "o-", color="#1f77b4")
    axes[0].set_xlabel(r"$n$ envs")
    axes[0].set_ylabel("Decision Hz (all envs)")
    axes[0].set_title("Throughput vs $n$")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ns, rt, "o-", color="#1f77b4", label="realtime factor")
    axes[1].plot(ns, eff, "s--", color="#ff7f0e", label="factor / $n$")
    axes[1].set_xlabel(r"$n$ envs")
    axes[1].set_ylabel("Speedup vs 20 Hz physics")
    axes[1].set_title("Realtime factor")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "bench_scaling.pdf")
    fig.savefig(FIG / "bench_scaling.png", dpi=160)
    plt.close(fig)

def roll_episode(config: dict, max_steps: int = 200) -> list[dict]:
    from bace import BaceEnv

    env = BaceEnv(config)
    env.reset(seed=int(config.get("env", {}).get("seed", 1)))
    frames = [env.state()]
    zeros = np.zeros(4, dtype=np.float32)
    for _ in range(max_steps):
        if not env.agents:
            break
        actions = {a: zeros for a in env.agents}
        env.step(actions)
        frames.append(env.state())
        if frames[-1].get("end") not in (None, "ongoing"):
            break
    env.close()
    return frames


def draw_tactical(ax, frames: list[dict], title: str, show_radar: bool = True) -> None:
    if not frames:
        return
    trails: dict[int, list[tuple[float, float]]] = {}
    for fr in frames:
        for f in fr.get("fighters", []):
            trails.setdefault(int(f["id"]), []).append(gdm_to_plot(f["pos"]))
    last = frames[-1]
    for f in last.get("fighters", []):
        fid = int(f["id"])
        color = "#4aa3ff" if int(f["team"]) == 0 else "#ff6b5a"
        xy = trails.get(fid, [])
        if len(xy) > 1:
            ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, lw=1.1, alpha=0.7)
        x, y = gdm_to_plot(f["pos"])
        hdg = math.radians(float(f["hdg"]))
        # heading 0 = north = +Y in this plot
        dx, dy = math.sin(hdg) * 2.2, math.cos(hdg) * 2.2
        ax.annotate(
            "",
            xy=(x + dx, y + dy),
            xytext=(x, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6),
        )
        ax.scatter([x], [y], c=color, s=18, zorder=4, alpha=1.0 if f.get("alive", True) else 0.3)
        if show_radar and f.get("alive", True):
            left = -RADAR_HFOV
            right = RADAR_HFOV
            angs = np.linspace(math.radians(float(f["hdg"]) + left), math.radians(float(f["hdg"]) + right), 40)
            rx = [x] + [x + math.sin(a) * RADAR_NM for a in angs] + [x]
            ry = [y] + [y + math.cos(a) * RADAR_NM for a in angs] + [y]
            ax.fill(rx, ry, color=color, alpha=0.06)
            ax.plot(rx, ry, color=color, lw=0.5, alpha=0.35)
    for m in last.get("missiles", []):
        mx, my = gdm_to_plot(m["pos"])
        ax.scatter([mx], [my], c="#ffd166" if m.get("pitbull") else "#c4b5fd", s=12, marker="s", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("East (NM)")
    ax.set_ylabel("North (NM)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.axhline(0, color="#8aa0b5", lw=0.4)
    ax.axvline(0, color="#8aa0b5", lw=0.4)


def plot_visualizer_views() -> dict:
    import matplotlib.pyplot as plt

    cases = {
        "wez_close": {
            "title": "WEZ shot, 16 NM head-on",
            "cfg": {
                "env": {"max_cycles": 80, "seed": 2, "stop_mission": False, "action_repeat": 20},
                "blue": {
                    "num_agents": 1,
                    "behavior": "fire_once",
                    "init_position": {"x": 0, "y": 25000, "z": 8},
                    "init_hdg": 0,
                    "target_position": {"x": 0, "y": 25000, "z": -8},
                },
                "red": {
                    "num_agents": 1,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -8},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 40},
                },
            },
        },
        "wez_far": {
            "title": "WEZ miss, 40 NM head-on",
            "cfg": {
                "env": {"max_cycles": 80, "seed": 3, "stop_mission": False},
                "blue": {
                    "num_agents": 1,
                    "behavior": "fire_once",
                    "init_position": {"x": 0, "y": 25000, "z": 20},
                    "init_hdg": 0,
                    "target_position": {"x": 0, "y": 25000, "z": -20},
                },
                "red": {
                    "num_agents": 1,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -20},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 50},
                },
            },
        },
        "bvr_4v4": {
            "title": "4$\\times$4 baseline vs duck",
            "cfg": {
                "env": {"max_cycles": 140, "seed": 4},
                "blue": {
                    "num_agents": 4,
                    "behavior": "baseline1",
                    "init_position": {"x": 0, "y": 25000, "z": 30},
                    "offset_pos": {"x": 4, "y": 0, "z": 4},
                },
                "red": {
                    "num_agents": 4,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "offset_pos": {"x": 4, "y": 0, "z": 4},
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
        "fsm_2v2": {
            "title": "2$\\times$2 baseline vs baseline",
            "cfg": {
                "env": {"max_cycles": 160, "seed": 7},
                "blue": {"num_agents": 2, "behavior": "baseline1"},
                "red": {
                    "num_agents": 2,
                    "behavior": "baseline1",
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
    }
    frames_out = {}
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.2))
    axes = axes.ravel()
    for ax, (key, spec) in zip(axes, cases.items()):
        frames = roll_episode(spec["cfg"], max_steps=180)
        frames_out[key] = {
            "steps": len(frames),
            "end": frames[-1].get("end") if frames else None,
            "n_fighters": len(frames[-1].get("fighters", [])) if frames else 0,
            "n_missiles": len(frames[-1].get("missiles", [])) if frames else 0,
        }
        draw_tactical(ax, frames, spec["title"])
    fig.tight_layout()
    fig.savefig(FIG / "visualizer_views.pdf")
    fig.savefig(FIG / "visualizer_views.png", dpi=170)
    plt.close(fig)
    return frames_out


def dump_behavior(frames: list[dict]) -> dict[str, Any]:
    from collections import Counter

    occ = Counter()
    ally = []
    ranges = []
    tracks = []
    fires = 0
    prev_m = None
    for fr in frames:
        blues = [f for f in fr.get("fighters", []) if int(f.get("team", 0)) == 0 and f.get("alive", True)]
        reds = [f for f in fr.get("fighters", []) if int(f.get("team", 0)) == 1 and f.get("alive", True)]
        for f in blues:
            occ[str(f.get("fsm", "Search"))] += 1
        if len(blues) >= 2:
            d = 0.0
            n = 0
            for i in range(len(blues)):
                for j in range(i + 1, len(blues)):
                    a, b = gdm_to_plot(blues[i]["pos"]), gdm_to_plot(blues[j]["pos"])
                    d += math.hypot(a[0] - b[0], a[1] - b[1])
                    n += 1
            if n:
                ally.append(d / n)
        if blues and reds:
            br = gdm_to_plot(blues[0]["pos"])
            rr = gdm_to_plot(reds[0]["pos"])
            ranges.append(math.hypot(br[0] - rr[0], br[1] - rr[1]))
        n_m = len(fr.get("missiles") or [])
        if prev_m is not None and n_m > prev_m:
            fires += n_m - prev_m
        prev_m = n_m
        tracks.append(1.0 if reds else 0.0)
    tot = sum(occ.values()) or 1
    return {
        "occupancy": {k: v / tot for k, v in occ.items()},
        "mean_ally_nm": float(np.mean(ally)) if ally else 0.0,
        "mean_range_nm": float(np.mean(ranges)) if ranges else 0.0,
        "fires": fires,
        "tracks_on": float(np.mean(tracks)) if tracks else 0.0,
        "end": frames[-1].get("end") if frames else None,
        "steps": len(frames),
    }


def plot_named_tactics() -> dict:
    import matplotlib.pyplot as plt

    named = {
        "hit": {
            "title": "Hit, 16 NM head-on",
            "cfg": {
                "env": {"max_cycles": 80, "seed": 2, "stop_mission": False},
                "blue": {
                    "num_agents": 1,
                    "behavior": "fire_once",
                    "init_position": {"x": 0, "y": 25000, "z": 8},
                    "target_position": {"x": 0, "y": 25000, "z": -8},
                },
                "red": {
                    "num_agents": 1,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -8},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 40},
                },
            },
        },
        "miss": {
            "title": "Miss, 40 NM head-on",
            "cfg": {
                "env": {"max_cycles": 80, "seed": 3, "stop_mission": False},
                "blue": {
                    "num_agents": 1,
                    "behavior": "fire_once",
                    "init_position": {"x": 0, "y": 25000, "z": 20},
                    "target_position": {"x": 0, "y": 25000, "z": -20},
                },
                "red": {
                    "num_agents": 1,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -20},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 50},
                },
            },
        },
        "4v4_duck": {
            "title": "4v4 baseline vs duck",
            "cfg": {
                "env": {"max_cycles": 140, "seed": 11},
                "blue": {"num_agents": 4, "behavior": "baseline1"},
                "red": {
                    "num_agents": 4,
                    "behavior": "duck",
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
        "4v4_fsm": {
            "title": "4v4 baseline vs baseline",
            "cfg": {
                "env": {"max_cycles": 160, "seed": 12},
                "blue": {"num_agents": 4, "behavior": "baseline1"},
                "red": {
                    "num_agents": 4,
                    "behavior": "baseline1",
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
        "striker_leak": {
            "title": "Red striker vs blue DCA (2v2)",
            "cfg": {
                "env": {"max_cycles": 160, "seed": 13},
                "blue": {"num_agents": 2, "behavior": "baseline1", "mission": "dca"},
                "red": {
                    "num_agents": 2,
                    "behavior": "duck",
                    "mission": "striker",
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
        "tracks_off": {
            "title": "2v2 share_tracks off",
            "cfg": {
                "env": {"max_cycles": 140, "seed": 14},
                "blue": {"num_agents": 2, "behavior": "baseline1", "share_tracks": False},
                "red": {
                    "num_agents": 2,
                    "behavior": "duck",
                    "share_tracks": False,
                    "init_position": {"x": 0, "y": 25000, "z": -30},
                    "init_hdg": 180,
                    "target_position": {"x": 0, "y": 25000, "z": 30},
                },
            },
        },
    }
    dumps = {}
    for key, spec in named.items():
        frames = roll_episode(spec["cfg"], max_steps=180)
        dumps[key] = dump_behavior(frames)
        fig, ax = plt.subplots(figsize=(5.2, 5.0))
        draw_tactical(ax, frames, spec["title"])
        fig.tight_layout()
        fig.savefig(FIG / f"tactical_{key}.pdf")
        fig.savefig(FIG / f"tactical_{key}.png", dpi=160)
        plt.close(fig)
    return dumps


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    FIG.mkdir(parents=True, exist_ok=True)
    (ROOT / "data").mkdir(parents=True, exist_ok=True)

    print("WEZ…")
    wez = run_wez_paper()
    plot_wez(wez)
    plot_wez_fit(wez)
    plot_wez_traces(wez)
    print(wez.get("summary"))

    print("FSM…")
    fsm = run_fsm_paper()
    plot_fsm(fsm)
    plot_fsm_occupancy(fsm)
    print(fsm.get("summary"))

    print("MARL…")
    marl = load_marl_paper()
    plot_marl(marl)
    plot_marl_eval(marl)
    print("marl jobs", marl.get("n_jobs"), "duck_2v2", marl.get("duck_2v2_learned"))

    print("Bench…")
    bench = load_json("bench.json")
    plot_bench(bench)
    print("bench n", len(bench.get("scaling") or []))

    print("Visualizer views…")
    views = plot_visualizer_views()
    dumps = plot_named_tactics()

    def slim_job(j):
        return {
            "algo": j.get("algo"),
            "opponent": j.get("opponent"),
            "agents": j.get("agents"),
            "seed": j.get("seed"),
            "episodes": j.get("episodes"),
            "first_mean": j.get("first_mean"),
            "last_mean": j.get("last_mean"),
            "eval": j.get("eval"),
            "eval_random": j.get("eval_random"),
            "eval_fire_once": j.get("eval_fire_once"),
            "beat_random": j.get("beat_random"),
            "kills_curve": j.get("kills_curve"),
            "share_tracks": j.get("share_tracks"),
            "red_mission": j.get("red_mission"),
        }

    slim = {
        "wez_summary": wez.get("summary"),
        "wez_cells": [
            {
                "range_nm": c["range_nm"],
                "altitude_ft": c["altitude_ft"],
                "aspect": c["aspect"],
                "n": c["n"],
                "hit_rate": c["hit_rate"],
                "analytic_rmax_nm": c["analytic_rmax_nm"],
            }
            for c in wez.get("cells", [])
        ],
        "wez_aspect_summaries": wez.get("aspect_summaries"),
        "wez_fits": wez.get("fits"),
        "wez_traces": wez.get("traces"),
        "fsm_summary": fsm.get("summary"),
        "fsm_history": fsm.get("history"),
        "fsm_elites": fsm.get("elites"),
        "fsm_params": fsm.get("params"),
        "marl_core": {
            "n_jobs": marl.get("n_jobs"),
            "jobs": [slim_job(j) for j in marl.get("jobs", [])],
            "duck_2v2_learned": marl.get("duck_2v2_learned"),
            "note": marl.get("note"),
        },
        "bench": {
            "cpu_count": bench.get("cpu_count"),
            "scaling": bench.get("scaling"),
            "pyo3_tax": bench.get("pyo3_tax"),
            "wez_wall": bench.get("wez_wall"),
            "fsm_wall": bench.get("fsm_wall"),
        },
        "views": views,
        "behavior": dumps,
    }
    (ROOT / "data" / "paper_results.json").write_text(json.dumps(slim, indent=2))
    print("wrote", ROOT / "data" / "paper_results.json")
    print("figures in", FIG)


if __name__ == "__main__":
    main()
