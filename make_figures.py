#!/usr/bin/env python3
"""Figures and the numbers table for the B-ACE 2.0 paper.

Reads the study artifacts that ``python -m bace.studies`` (in B-ACE-2.0) writes to
``runs/paper/``, copied into ``data/``:

    missile.json wez.json crossplay.json ablation.json landscape.json defend.json
    bench.json rl_eval.json rl_train.json

No simulation runs here. Output: ``figures/*.pdf`` and ``data/numbers.json`` (every
number quoted in the text).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"

# Palette (validated categorical order, light mode) and text inks.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
SEQ = ["#9cc3f0", "#5a9be6", "#2a78d6", "#1a56a3"]  # one hue, light -> dark

# Fixed colour per entity, never per rank.
PILOT_COLOR = {
    "rl": BLUE,
    "ref_aggressive": ORANGE,
    "ref_balanced": AQUA,
    "ref_conservative": YELLOW,
    "fsm_b1": MAGENTA,
    "fire_once": MUTED,
    "duck": MUTED,
}
PILOT_LABEL = {
    "rl": "PPO (learned)",
    "ref_aggressive": "Aggressive",
    "ref_balanced": "Balanced",
    "ref_conservative": "Conservative",
    "fsm_b1": "FSM-B1",
    "fire_once": "Fire-once",
    "duck": "Duck",
}
ASPECT_COLOR = {"head": BLUE, "beam": ORANGE, "tail": AQUA}

NUM: dict[str, Any] = {}


def load(name: str) -> dict[str, Any]:
    p = DATA / name
    return json.loads(p.read_text()) if p.is_file() else {}


def style() -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 1.6,
            "figure.dpi": 150,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=170)
    import matplotlib.pyplot as plt

    plt.close(fig)


def wilson(k: float, n: float) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", [RED, "#f4c9c7", "#f2f1ee", "#c4daf4", BLUE])


# ------------------------------------------------------------------ missile


def fig_missile(d: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    tr = d["traces"]
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.6))
    hot = [c for c in tr if c["model"] == "GenericMraam" and c["target"] == "hot"]
    for i, c in enumerate(hot):
        s = c["samples"]
        col = SEQ[i + 1]
        lab = f"{c['launch_alt_ft']/1000:.0f} kft: {c['end']} at {c['tof_s']:.0f} s"
        ax[0].plot([x["t"] for x in s], [x["mach"] for x in s], color=col, label=lab)
        ax[1].plot(
            [x["downrange_nm"] for x in s], [x["alt_ft"] / 1000 for x in s], color=col
        )
        NUM[f"missile_peak_mach_{int(c['launch_alt_ft']/1000)}k"] = max(x["mach"] for x in s)
        NUM[f"missile_tof_{int(c['launch_alt_ft']/1000)}k"] = c["tof_s"]
        NUM[f"missile_end_{int(c['launch_alt_ft']/1000)}k"] = c["end"]
    leg = [c for c in tr if c["model"] == "Legacy"][0]
    a_25 = 309.0  # speed of sound at 25 kft, m/s
    ax[0].plot(
        [x["t"] for x in leg["samples"]],
        [x["speed_ms"] / a_25 for x in leg["samples"]],
        color=MUTED,
        ls="--",
        lw=1.2,
        label="B-ACE 2.0 missile (25 kft)",
    )
    ax[0].axhline(1.0, color=MUTED, lw=0.6)
    ax[0].set_xlabel("time of flight (s)")
    ax[0].set_ylabel("Mach")
    ax[0].set_title("Speed: boost, sustain, then coast")
    ax[0].legend(loc="lower left", bbox_to_anchor=(0.12, 0.0), fontsize=6.8)
    ax[1].set_xlabel("downrange (NM)")
    ax[1].set_ylabel("altitude (kft)")
    ax[1].set_title("Lofted trajectory, target 30 NM head-on")
    for c in [c for c in tr if c["target"] == "cold"]:
        NUM[f"missile_cold_{int(c['range_nm'])}nm"] = c["end"]
    save(fig, "missile")


# ------------------------------------------------------------------ WEZ


def fig_wez_table(d: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    t = d["table"]
    sa, ta = t["shooter_alt_ft"], t["target_alt_ft"]
    asp, ob = t["target_aspect_deg"], t["off_boresight_deg"]
    R = np.asarray(t["rmax_nm"], dtype=float).reshape(len(sa), len(ta), len(asp), len(ob))
    N = np.asarray(t["rnez_nm"], dtype=float).reshape(R.shape)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
    im = ax[0].imshow(R[:, :, 0, 0], origin="lower", cmap="Blues", aspect="auto")
    ax[0].set_xticks(range(len(ta)), [f"{x/1000:.0f}" for x in ta])
    ax[0].set_yticks(range(len(sa)), [f"{x/1000:.0f}" for x in sa])
    ax[0].set_xlabel("target altitude (kft)")
    ax[0].set_ylabel("shooter altitude (kft)")
    ax[0].set_title("Head-on $R_{max}$ (NM)")
    ax[0].grid(False)
    for i in range(len(sa)):
        for j in range(len(ta)):
            v = R[i, j, 0, 0]
            ax[0].text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                       color="white" if v > np.nanmax(R[:, :, 0, 0]) * 0.6 else INK)
    for k, alt in enumerate([5000, 25000, 45000]):
        i = sa.index(alt)
        col = SEQ[k + 1]
        ax[1].plot(asp, R[i, i, :, 0], color=col, label=f"{alt/1000:.0f} kft $R_{{max}}$")
        ax[1].plot(asp, N[i, i, :, 0], color=col, ls="--", lw=1.2, label=f"{alt/1000:.0f} kft $R_{{NEZ}}$")
    ax[1].set_xlabel("target aspect (deg, 0 = hot)")
    ax[1].set_ylabel("range (NM)")
    ax[1].set_title("Co-altitude envelope vs target aspect")
    ax[1].set_xticks([0, 45, 90, 135, 180])
    ax[1].legend(ncol=1, fontsize=6.8, loc="upper right")
    i25 = sa.index(25000)
    NUM["wez_head_rmax"] = {f"{a/1000:.0f}k": float(R[sa.index(a), sa.index(a), 0, 0]) for a in sa}
    NUM["wez_tail_rmax"] = {f"{a/1000:.0f}k": float(R[sa.index(a), sa.index(a), -1, 0]) for a in sa}
    NUM["wez_head_rnez_25k"] = float(N[i25, i25, 0, 0])
    NUM["wez_axes"] = {"shooter_alt": len(sa), "target_alt": len(ta), "aspect": len(asp), "off_boresight": len(ob)}
    save(fig, "wez_table")


def fig_wez_validation(d: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    val = d["validation"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True, sharey=True)
    ratios = {}
    for r, model in enumerate(["generic_mraam", "legacy"]):
        cells = val[model]["cells"]
        for c_i, alt in enumerate([10000.0, 25000.0, 40000.0]):
            ax = axes[r, c_i]
            for asp in ("head", "beam", "tail"):
                cs = sorted(
                    [c for c in cells if c["altitude_ft"] == alt and c["aspect"] == asp],
                    key=lambda c: c["range_nm"],
                )
                x = [c["range_nm"] for c in cs]
                y = [c["hit_rate"] for c in cs]
                ax.plot(x, y, color=ASPECT_COLOR[asp], lw=1.4, label=asp)
                rm = float(np.median([c["analytic_rmax_nm"] for c in cs]))
                if rm < 50:
                    ax.axvline(rm, color=ASPECT_COLOR[asp], ls=":", lw=1.0)
                # outer 50% crossing
                p50 = None
                for a, b in zip(cs[::-1][1:], cs[::-1][:-1]):
                    if a["hit_rate"] >= 0.5 > b["hit_rate"]:
                        f = (a["hit_rate"] - 0.5) / max(a["hit_rate"] - b["hit_rate"], 1e-9)
                        p50 = a["range_nm"] + f * (b["range_nm"] - a["range_nm"])
                        break
                if p50 is not None and rm < 50:
                    ratios[f"{model}|{alt:.0f}|{asp}"] = {"p50": p50, "rmax": rm, "ratio": p50 / rm}
            ax.set_title(f"{'Generic' if model == 'generic_mraam' else 'B-ACE 2.0'} missile, {alt/1000:.0f} kft", fontsize=8)
            if c_i == 0:
                ax.set_ylabel("hit rate")
            if r == 1:
                ax.set_xlabel("launch range (NM)")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    NUM["wez_validation"] = ratios
    g = [v["ratio"] for k, v in ratios.items() if k.startswith("generic")]
    NUM["wez_ratio_generic"] = [min(g), max(g)] if g else []
    save(fig, "wez_validation")


# ------------------------------------------------------------------ scripted


def _pk_bins(raw: list[list[float]], edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray([s[1] / s[2] for s in raw if s[2] > 0 and s[3] >= 0])
    h = np.asarray([s[3] for s in raw if s[2] > 0 and s[3] >= 0])
    mid, p, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (r >= a) & (r < b)
        n = int(m.sum())
        if n < 15:
            continue
        k = float(h[m].sum())
        mid.append((a + b) / 2)
        p.append(k / n)
        l, u = wilson(k, n)
        lo.append(l)
        hi.append(u)
    return np.asarray(mid), np.asarray(p), np.asarray(lo), np.asarray(hi)


def fig_pk(cross: dict[str, Any], ablation: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    cells = cross["1v1"]
    raw = [s for c in cells.values() for s in c.get("raw_shots", [])]
    edges = np.arange(0.2, 1.11, 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.7})
    mid, p, lo, hi = _pk_bins(raw, edges)
    ax[0].fill_between(mid, lo, hi, color=BLUE, alpha=0.18, lw=0)
    ax[0].plot(mid, p, "o-", color=BLUE, ms=4, label="target defends (missile warning)")
    ax[0].set_xlabel("launch range / $R_{max}$")
    ax[0].set_ylabel("hit rate")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Hit rate vs launch range, scripted 1v1 duels")
    NUM["pk_bins"] = {f"{m:.2f}": float(v) for m, v in zip(mid, p)}
    NUM["pk_n_shots"] = len(raw)
    # per pilot
    for name in cross["pilots"]:
        rs = [s for k, c in cells.items() if k.startswith(name + "|") for s in c.get("raw_shots", []) if s[0] == 0]
        rr = [s[1] / s[2] for s in rs if s[2] > 0]
        hh = [s[3] for s in rs if s[3] >= 0]
        if rr:
            ax[0].plot(np.median(rr), np.mean(hh), "D", color=PILOT_COLOR[name], ms=6,
                       markeredgecolor="white", mew=1.2)
            ax[0].annotate(PILOT_LABEL[name], (np.median(rr), np.mean(hh)), textcoords="offset points",
                           xytext=(-8, 16) if name == "fsm_b1" else (6, 4), fontsize=7, color=INK2)
            NUM[f"pilot_shot_{name}"] = {"range_over_rmax_median": float(np.median(rr)), "hit_rate": float(np.mean(hh)), "n": len(rr)}
    # ablation hit rate & draws by condition
    conds = list(ablation["conditions"])
    P = ablation["pilots"]
    labels = {
        "full": "Full model",
        "no_datalink": "No datalink",
        "no_warning": "No missile warning",
        "constant_speed": "No energy loss",
        "legacy_missile": "2.0 missile",
        "bace_2_0": "B-ACE 2.0 physics",
    }
    hit = [np.mean([ablation[c][f"{b}|{r}"]["blue_shots"]["hit_rate"] for b in P for r in P]) for c in conds]
    lose = [np.mean([ablation[c][f"{b}|{r}"]["kills"][0] + ablation[c][f"{b}|{r}"]["deaths"][0] for b in P for r in P]) for c in conds]
    y = np.arange(len(conds))[::-1]
    ax[1].barh(y + 0.18, hit, height=0.34, color=BLUE, label="hit rate")
    ax[1].barh(y - 0.18, np.asarray(lose) / 2, height=0.34, color=ORANGE, label="aircraft lost / episode / side")
    ax[1].set_yticks(y, [labels[c] for c in conds])
    ax[1].set_xlim(0, 1)
    ax[1].set_title("Same pilots, different physics")
    ax[1].legend(loc="upper center", bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=7)
    ax[1].grid(axis="y", visible=False)
    NUM["ablation_hit"] = dict(zip(conds, map(float, hit)))
    NUM["ablation_losses_per_side"] = dict(zip(conds, [float(x) / 2 for x in lose]))
    save(fig, "pk_and_ablation")


def fig_crossplay(cross: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    P = cross["pilots"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    cmap = diverging()
    for ax, size in zip(axes, ("1v1", "2v2")):
        M = np.array([[cross[size][f"{b}|{r}"]["net"][0] for r in P] for b in P])
        lim = 0.8
        ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim)
        for i, b in enumerate(P):
            for j, r in enumerate(P):
                c = cross[size][f"{b}|{r}"]
                ax.text(j, i, f"{c['net'][0]:+.2f}\n{c['win'][0]:.2f}/{c['lose'][0]:.2f}",
                        ha="center", va="center", fontsize=6.5, color=INK)
        ax.set_xticks(range(len(P)), [PILOT_LABEL[p] for p in P], rotation=25, ha="right")
        ax.set_yticks(range(len(P)), [PILOT_LABEL[p] for p in P])
        ax.set_title(f"{size}: blue (row) vs red (column)")
        ax.grid(False)
        ax.set_xlabel("red pilot")
        if size == "1v1":
            ax.set_ylabel("blue pilot")
        NUM[f"crossplay_{size}"] = {
            f"{b}|{r}": {k: cross[size][f"{b}|{r}"][k] for k in ("net", "win", "lose", "draw")}
            for b in P
            for r in P
        }
        NUM[f"crossplay_{size}_row_mean"] = {b: float(M[i].mean()) for i, b in enumerate(P)}
        NUM[f"crossplay_{size}_draw"] = float(np.mean([cross[size][f"{b}|{r}"]["draw"] for b in P for r in P]))
    save(fig, "crossplay")


def fig_landscape(d: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    ds, lb = d["d_shot"], d["l_break"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    cmap = diverging()
    for ax, c, title in zip(axes, ("bace_2_0", "full"), ("B-ACE 2.0 physics", "Full model")):
        G = np.array([[x["net"] for x in row] for row in d[c]])
        H = np.array([[x["hit_rate"] for x in row] for row in d[c]])
        ax.imshow(G, origin="lower", cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        for i in range(len(ds)):
            for j in range(len(lb)):
                ax.text(j, i, f"{G[i, j]:+.2f}", ha="center", va="center", fontsize=6, color=INK)
        best = np.unravel_index(np.argmax(G), G.shape)
        ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
        ax.set_xticks(range(len(lb)), [f"{x:.1f}" for x in lb])
        ax.set_yticks(range(len(ds)), [f"{x:.1f}" for x in ds])
        ax.set_xlabel("$l_{break}$ (threat factor that triggers the break)")
        ax.set_title(f"{title}: net exchange vs Balanced")
        ax.grid(False)
        NUM[f"landscape_{c}"] = {
            "best": {"d_shot": ds[best[0]], "l_break": lb[best[1]], "net": float(G[best])},
            "mean_hit": float(np.nanmean(H)),
            "positive_cells": int((G > 0.05).sum()),
            "cells": int(G.size),
        }
    axes[0].set_ylabel("$d_{shot}$ (launch at this fraction of $R_{max}$)")
    save(fig, "landscape")


# ------------------------------------------------------------------ RL


def _smooth(x: np.ndarray, k: int = 15) -> np.ndarray:
    if len(x) < k:
        return x
    w = np.ones(k) / k
    return np.convolve(np.pad(x, (k // 2, k - 1 - k // 2), mode="edge"), w, mode="valid")


def fig_rl_curves(train: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    opps = ["duck", "ref_aggressive", "ref_balanced", "ref_conservative"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    for ax, size in zip(axes, ("1v1", "2v2")):
        runs = [r for r in train.get("runs", []) if r["agents"] == int(size[0])]
        if not runs:
            continue
        grid = np.linspace(0, min(r["history"][-1]["env_steps"] for r in runs), 120)
        for o in opps:
            curves = []
            for r in runs:
                xs, ys = [], []
                for h in r["history"]:
                    v = h["by_opponent"].get(o)
                    if v:
                        xs.append(h["env_steps"])
                        ys.append(v["win"] - v["lose"])
                ys = _smooth(np.asarray(ys), 25)
                curves.append(np.interp(grid, xs, ys))
            C = np.asarray(curves)
            col = PILOT_COLOR.get(o, MUTED)
            ax.plot(grid / 1e6, C.mean(0), color=col, label=f"vs {PILOT_LABEL[o]}")
            ax.fill_between(grid / 1e6, C.min(0), C.max(0), color=col, alpha=0.15, lw=0)
        for r in runs:
            if r.get("selected_steps"):
                ax.axvline(r["selected_steps"] / 1e6, color=MUTED, lw=0.6, ls=":")
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("agent decisions (millions)")
        ax.set_title(f"{size} defence, {len(runs)} seeds (mean, range)")
    axes[0].set_ylabel("win rate - loss rate (training)")
    axes[0].legend(loc="lower right", fontsize=7)
    save(fig, "rl_curves")


def fig_rl_eval(rl: dict[str, Any], defend: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    strikers = ["ref_aggressive", "ref_balanced", "ref_conservative"]
    defenders = ["rl", "ref_aggressive", "ref_balanced", "ref_conservative", "fsm_b1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, size in zip(axes, ("1v1", "2v2")):
        agents = int(size[0])
        pols = [p for p in rl.get("policies", {}).values() if p["agents"] == agents]
        x = np.arange(len(strikers))
        w = 0.16
        for k, dname in enumerate(defenders):
            if dname == "rl":
                if not pols:
                    continue
                vals = np.array([[p["eval"][s]["win"][0] - p["eval"][s]["lose"][0] for s in strikers] for p in pols])
                m = vals.mean(0)
                err = np.vstack([m - vals.min(0), vals.max(0) - m])
                ax.bar(x + (k - 2) * w, m, w * 0.92, color=PILOT_COLOR[dname], label=PILOT_LABEL[dname],
                       yerr=err, ecolor=INK2, capsize=2, error_kw={"lw": 0.8})
                NUM[f"rl_eval_{size}"] = {
                    s: {
                        "win_mean": float(np.mean([p["eval"][s]["win"][0] for p in pols])),
                        "lose_mean": float(np.mean([p["eval"][s]["lose"][0] for p in pols])),
                        "net_mean": float(np.mean([p["eval"][s]["net"][0] for p in pols])),
                        "per_seed_win": [p["eval"][s]["win"][0] for p in pols],
                        "per_seed_lose": [p["eval"][s]["lose"][0] for p in pols],
                    }
                    for s in strikers + ["duck"]
                }
            else:
                cells = defend[size]
                m = np.array([cells[f"{dname}|{s}"]["win"][0] - cells[f"{dname}|{s}"]["lose"][0] for s in strikers])
                ax.bar(x + (k - 2) * w, m, w * 0.92, color=PILOT_COLOR[dname], label=PILOT_LABEL[dname])
                NUM[f"defend_{size}_{dname}"] = {
                    s: {"win": cells[f"{dname}|{s}"]["win"][0], "lose": cells[f"{dname}|{s}"]["lose"][0]}
                    for s in strikers + ["duck"]
                }
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_xticks(x, [f"vs\n{PILOT_LABEL[s]}" for s in strikers])
        ax.set_title(f"{size} defence, held-out episodes")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("win rate - loss rate")
    axes[0].legend(fontsize=7, loc="upper center", bbox_to_anchor=(1.1, -0.22), ncol=5)
    save(fig, "rl_eval")


def _strategy(shots: list[list[float]], episodes: int) -> dict[str, Any]:
    """Shot statistics from raw shot rows (both teams, with episode index)."""
    blue = [x for x in shots if x[0] == 0 and x[2] > 0]
    red = [x for x in shots if x[0] == 1]
    first: dict[int, list[float]] = {}
    for x in blue:
        first.setdefault(int(x[9]), x)
    hits = [x for x in blue if x[3] == 1]
    done = [x for x in blue if x[3] >= 0]
    fs = list(first.values())
    return {
        "shots_per_episode": len(blue) / episodes,
        "hit_rate": len(hits) / max(1, len(done)),
        "range_over_rmax_median": float(np.median([x[1] / x[2] for x in blue])) if blue else float("nan"),
        "first_shot_range_over_rmax": float(np.median([x[1] / x[2] for x in fs])) if fs else float("nan"),
        "first_shot_hit": float(np.mean([x[3] == 1 for x in fs])) if fs else float("nan"),
        "hit_target_aspect": float(np.mean([x[5] for x in hits])) if hits else float("nan"),
        "hit_tof_s": float(np.mean([x[8] for x in hits])) if hits else float("nan"),
        "enemy_shots_per_episode": len(red) / episodes,
    }


def fig_rl_tactics(rl: dict[str, Any], defend: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    strikers = ["ref_aggressive", "ref_balanced", "ref_conservative"]
    pols = sorted(
        [p for p in rl.get("policies", {}).values() if p["agents"] == 1], key=lambda p: p["seed"]
    )
    if not pols:
        return
    entities: list[tuple[str, str, str, list[list[float]]]] = []
    seed_cols = [SEQ[3], SEQ[2], SEQ[1]]
    for k, p in enumerate(pols):
        rows = []
        for i, st in enumerate(strikers):
            rows += [x[:9] + [x[9] + 1000 * i] for x in p["eval"][st].get("raw_shots", [])]
        entities.append((f"rl_s{p['seed']}", f"PPO seed {p['seed']}", seed_cols[k % 3], rows))
    for dname in ("ref_aggressive", "ref_balanced", "ref_conservative"):
        rows = []
        for i, st in enumerate(strikers):
            rows += [x[:9] + [x[9] + 1000 * i] for x in defend["1v1"][f"{dname}|{st}"].get("raw_shots", [])]
        entities.append((dname, PILOT_LABEL[dname], PILOT_COLOR[dname], rows))
    episodes = 200 * len(strikers)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.55})
    bins = np.linspace(0, 1.2, 25)
    for key, label, col, rows in entities:
        r = np.asarray([x[1] / x[2] for x in rows if x[0] == 0 and x[2] > 0])
        NUM[f"tactics_{key}"] = _strategy(rows, episodes)
        if r.size == 0:
            continue
        h, _ = np.histogram(r, bins=bins)
        axes[0].step(bins[:-1], h / episodes, where="post", color=col, label=label,
                     lw=1.8 if key.startswith("rl") else 1.1, ls="-" if key.startswith("rl") else "--")
    axes[0].set_xlabel("launch range / $R_{max}$")
    axes[0].set_ylabel("missiles per episode per bin")
    axes[0].set_title("Where 1v1 defenders shoot")
    axes[0].legend(fontsize=6.8, ncol=2)
    names = [e[0] for e in entities]
    y = np.arange(len(names))[::-1]
    for yy, (key, label, col, _) in zip(y, entities):
        st = NUM[f"tactics_{key}"]
        axes[1].scatter(st["first_shot_range_over_rmax"], yy, color=col, s=36, marker="o", zorder=3,
                        edgecolor="white", lw=1)
        axes[1].scatter(st["hit_rate"], yy, color=col, s=40, marker="D", zorder=3, edgecolor="white", lw=1)
        axes[1].text(1.04, yy, f"{st['enemy_shots_per_episode']:.2f}", va="center", fontsize=7, color=INK2)
    axes[1].set_yticks(y, [e[1] for e in entities])
    axes[1].set_xlim(-0.08, 1.15)
    axes[1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[1].set_ylim(-0.6, len(names) - 0.4)
    axes[1].set_title("first shot at / $R_{max}$ (o), hit rate (\u25c6),\nenemy shots per episode (right)", fontsize=8)
    axes[1].grid(axis="y", visible=False)
    save(fig, "rl_tactics")


# ------------------------------------------------------------------ bench & architecture


def fig_bench(b: dict[str, Any], rl_train: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    for c in ("bace_2_0", "full"):
        rows = b.get(c, [])
        NUM[f"bench_{c}"] = {str(r["n_envs"]): r["decision_hz"] for r in rows}
    NUM["bench_threads"] = b.get("host_threads")
    return
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for c, col, lab in (("bace_2_0", MUTED, "B-ACE 2.0 physics"), ("full", BLUE, "Full model")):
        rows = b.get(c, [])
        ax.plot([r["n_envs"] for r in rows], [r["decision_hz"] / 1e3 for r in rows], "o-", color=col, ms=4, label=lab)
        NUM[f"bench_{c}"] = {str(r["n_envs"]): r["decision_hz"] for r in rows}
    ax.set_xscale("log", base=2)
    ax.set_xlabel("parallel games")
    ax.set_ylabel("decisions / s (thousands)")
    ax.set_title(f"Rust batch runner, {b.get('host_threads')} CPU threads")
    ax.legend(fontsize=7)
    NUM["bench_threads"] = b.get("host_threads")
    save(fig, "bench")


def fig_architecture() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.1",
                                    linewidth=0.8, edgecolor=MUTED, facecolor=fc))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, color=INK)

    box(0.2, 2.75, 3.0, 1.15, "Python\nPettingZoo / Gymnasium\nPPO, studies, recordings", "#e3eefb")
    box(3.5, 2.75, 3.0, 1.15, "B-ACE Lab (browser)\nsimulate, evaluate, WEZ,\nFSM search, replay", "#fde8de")
    box(6.8, 2.75, 3.0, 1.15, "bace-server (Axum)\nserves the Lab and\nrecorded runs", "#f2f1ee")
    box(0.2, 1.5, 3.0, 0.95, "bace-py (PyO3)\nNativeEnv / NativeVecEnv", "#e2f5ee")
    box(3.5, 1.5, 3.0, 0.95, "bace-wasm\nWebAssembly, JSON C ABI", "#e2f5ee")
    box(6.8, 1.5, 3.0, 0.95, "bace-record\nepisode JSON", "#e2f5ee")
    box(0.2, 0.2, 6.3, 1.0,
        "bace-core: atmosphere, aircraft energy, radar, datalink,\nmissile (motor, drag, PN, seeker), fly-out WEZ table,\n"
        "missile warning, scripted pilots, rewards, shot log",
        "#fbf3dc")
    box(6.8, 0.2, 3.0, 1.0, "bace-vec\nrayon batches,\nrecipes", "#fbf3dc")
    for x in (1.7, 5.0, 8.3):
        ax.annotate("", xy=(x, 2.75), xytext=(x, 2.45), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
        ax.annotate("", xy=(x, 1.5), xytext=(x, 1.2), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8))
    save(fig, "architecture")


def write_macros() -> None:
    """numbers.tex: every number quoted in the text, as a LaTeX macro."""
    m: dict[str, str] = {}

    def f2(x: float) -> str:
        return f"{x:.2f}"

    m["MachLow"] = f"{NUM['missile_peak_mach_10k']:.1f}"
    m["MachHigh"] = f"{NUM['missile_peak_mach_40k']:.1f}"
    m["TofHigh"] = f"{NUM['missile_tof_40k']:.0f}"
    m["RmaxFive"] = f"{NUM['wez_head_rmax']['5k']:.0f}"
    m["RmaxFortyFive"] = f"{NUM['wez_head_rmax']['45k']:.0f}"
    ax = NUM["wez_axes"]
    m["WezCells"] = str(ax["shooter_alt"] * ax["target_alt"] * ax["aspect"] * ax["off_boresight"])
    lo, hi = NUM["wez_ratio_generic"]
    m["WezRatioLo"] = f2(lo)
    m["WezRatioHi"] = f2(hi)
    pk = NUM["pk_bins"]
    m["PkLow"] = f2(pk["0.35"])
    m["PkMid"] = f2(pk["0.55"])
    m["PkShots"] = f"{NUM['pk_n_shots']:,}".replace(",", "{,}")
    m["PilotAggHit"] = f2(NUM["pilot_shot_ref_aggressive"]["hit_rate"])
    m["PilotBalHit"] = f2(NUM["pilot_shot_ref_balanced"]["hit_rate"])
    m["PilotConHit"] = f2(NUM["pilot_shot_ref_conservative"]["hit_rate"])
    m["DrawFull"] = f"{100 * NUM['crossplay_1v1_draw']:.0f}\\%"
    m["CrossBal"] = f"{NUM['crossplay_1v1_row_mean']['ref_balanced']:+.2f}"
    m["CrossTwoCon"] = f"{NUM['crossplay_2v2_row_mean']['ref_conservative']:+.2f}"
    m["CrossTwoBal"] = f"{NUM['crossplay_2v2_row_mean']['ref_balanced']:+.2f}"
    h = NUM["ablation_hit"]
    m["HitFull"] = f2(h["full"])
    m["HitNoWarn"] = f2(h["no_warning"])
    m["HitConst"] = f2(h["constant_speed"])
    m["HitOld"] = f2(h["bace_2_0"])
    m["LandOldBest"] = f2(NUM["landscape_bace_2_0"]["best"]["net"])
    m["BenchFullOne"] = f"{NUM['bench_full']['1'] / 1e3:.0f}"
    m["BenchOldOne"] = f"{NUM['bench_bace_2_0']['1'] / 1e3:.0f}"
    rl1 = NUM.get("rl_eval_1v1", {})
    if rl1:
        m["RLOneWinBal"] = f2(rl1["ref_balanced"]["win_mean"])
    for size in ("1v1", "2v2"):
        ev = NUM.get(f"rl_eval_{size}")
        if not ev:
            continue
        tag = "One" if size == "1v1" else "Two"
        for st, nm in (("ref_aggressive", "Agg"), ("ref_balanced", "Bal"), ("ref_conservative", "Con")):
            ws = ev[st]["per_seed_win"]
            ls = ev[st]["per_seed_lose"]
            m[f"RL{tag}Win{nm}"] = f2(ev[st]["win_mean"])
            m[f"RL{tag}WinRange{nm}"] = f"{min(ws):.2f}--{max(ws):.2f}"
            m[f"RL{tag}LoseMax{nm}"] = f2(max(ls))
        allw = [w for st in ("ref_aggressive", "ref_balanced", "ref_conservative") for w in ev[st]["per_seed_win"]]
        alll = [x for st in ("ref_aggressive", "ref_balanced", "ref_conservative") for x in ev[st]["per_seed_lose"]]
        m[f"RL{tag}WinMin"] = f2(min(allw))
        m[f"RL{tag}WinMax"] = f2(max(allw))
        m[f"RL{tag}LoseMax"] = f2(max(alll))
    best_scripted = {}
    for size in ("1v1", "2v2"):
        vals = [
            v[st]["win"]
            for k, v in NUM.items()
            if k.startswith(f"defend_{size}_")
            for st in ("ref_aggressive", "ref_balanced", "ref_conservative")
        ]
        if vals:
            best_scripted[size] = max(vals)
            m[f"Scripted{'One' if size == '1v1' else 'Two'}WinMax"] = f2(max(vals))
    for key in ("rl_s0", "rl_s1", "rl_s2", "ref_aggressive", "ref_balanced", "ref_conservative"):
        t = NUM.get(f"tactics_{key}")
        if not t:
            continue
        tag = {"rl_s0": "SZero", "rl_s1": "SOne", "rl_s2": "STwo", "ref_aggressive": "Agg",
               "ref_balanced": "Bal", "ref_conservative": "Con"}[key]
        m[f"Tac{tag}First"] = f2(t["first_shot_range_over_rmax"])
        m[f"Tac{tag}FirstHit"] = f2(t["first_shot_hit"])
        m[f"Tac{tag}Hit"] = f2(t["hit_rate"])
        m[f"Tac{tag}Shots"] = f"{t['shots_per_episode']:.1f}"
        m[f"Tac{tag}Enemy"] = f2(t["enemy_shots_per_episode"])
        m[f"Tac{tag}Aspect"] = f"{t['hit_target_aspect']:.0f}"
        m[f"Tac{tag}Tof"] = f"{t['hit_tof_s']:.0f}"
    tr = [r for r in NUM.get("rl_train", []) if r.get("agents") == 1]
    if tr:
        m["RLTimeOne"] = f"{np.median([r['train_time_s'] for r in tr]) / 60:.0f}"
        m["RLThroughput"] = f"{np.median([r['env_steps'] / r['train_time_s'] for r in tr]) / 1e3:.0f}{{,}}000"
    lines = [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(m.items())]
    (ROOT / "numbers.tex").write_text("% generated by make_figures.py\n" + "\n".join(lines) + "\n")


def main() -> None:
    import matplotlib.pyplot as plt  # noqa: F401

    style()
    FIG.mkdir(exist_ok=True)
    fig_architecture()
    missile = load("missile.json")
    if missile:
        fig_missile(missile)
    wez = load("wez.json")
    if wez:
        fig_wez_table(wez)
        fig_wez_validation(wez)
    cross, abl = load("crossplay.json"), load("ablation.json")
    if cross and abl:
        fig_pk(cross, abl)
        fig_crossplay(cross)
        NUM["ablation_rank"] = {
            c: sorted(abl["pilots"], key=lambda b: -np.mean([abl[c][f"{b}|{r}"]["net"][0] for r in abl["pilots"]]))
            for c in abl["conditions"]
        }
        NUM["ablation_score"] = {
            c: {b: float(np.mean([abl[c][f"{b}|{r}"]["net"][0] for r in abl["pilots"]])) for b in abl["pilots"]}
            for c in abl["conditions"]
        }
        NUM["ablation_draw"] = {
            c: float(np.mean([abl[c][f"{b}|{r}"]["draw"] for b in abl["pilots"] for r in abl["pilots"]]))
            for c in abl["conditions"]
        }
    land = load("landscape.json")
    if land:
        fig_landscape(land)
    defend = load("defend.json")
    rl = load("rl_eval.json")
    train = load("rl_train.json")
    if train:
        fig_rl_curves(train)
        NUM["rl_train"] = [
            {k: r.get(k) for k in ("name", "agents", "seed", "env_steps", "train_time_s", "selected_steps")}
            for r in train["runs"]
        ]
    if defend:
        fig_rl_eval(rl, defend)
        fig_rl_tactics(rl, defend)
    bench = load("bench.json")
    if bench:
        fig_bench(bench, train)
    (DATA / "numbers.json").write_text(json.dumps(NUM, indent=1, default=float) + "\n")
    write_macros()
    print("figures:", sorted(p.name for p in FIG.glob("*.pdf")))


if __name__ == "__main__":
    main()
