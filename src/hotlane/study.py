"""Train the RL controller, evaluate all controllers on held-out seeds, run the drift experiment."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .controllers import PID, DensityLookup, QLearning, TimeOfDay
from .drift import PageHinkley
from .model import MPH, Corridor, CorridorParams

UPDATE_S = 180.0  # toll update interval (3 min)


def episode(p: CorridorParams, ctrl, seed: int, *, learn: bool = False, vot_shift_at_s: float | None = None,  # type: ignore[no-untyped-def]
            vot_shift: float = 1.0) -> Corridor:
    c = Corridor(p, seed, vot_shift_at_s=vot_shift_at_s, vot_shift=vot_shift)
    ctrl.reset()
    steps_per = int(UPDATE_S / p.dt_s)
    n = int(p.duration_s / p.dt_s)
    toll = ctrl.toll(c.observe())
    prev_toll = toll
    flow_acc = 0.0
    for i in range(n):
        rec = c.step(toll)
        flow_acc += rec["flow_hot"] + rec["flow_gp"]
        if (i + 1) % steps_per == 0:
            obs = c.observe()
            if learn and isinstance(ctrl, QLearning):
                ctrl.learn(obs, QLearning.reward(obs, flow_acc / steps_per * 3600.0, toll - prev_toll))
            prev_toll = toll
            toll = ctrl.toll(obs)
            flow_acc = 0.0
    return c


def metrics(c: Corridor) -> dict[str, float]:
    p = c.p
    lg = {k: np.asarray(v) for k, v in c.state.log.items()}
    peak = (lg["t"] >= 3600.0) & (lg["t"] < 3 * 3600.0)
    ok = lg["hot_speed"] >= 45.0 * MPH
    return {
        "hot_reliability_peak": float(ok[peak].mean()),
        "hot_throughput_vph": float(lg["flow_hot"][peak].mean() * 3600.0),
        "gp_throughput_vph": float(lg["flow_gp"][peak].mean() * 3600.0),
        "gp_mean_speed_mph": float(lg["gp_speed"][peak].mean() / MPH),
        "entry_queue_veh_h": float(lg["queue"].sum() * p.dt_s / 3600.0),
        "revenue_usd": float(lg["revenue"].sum()),
        "mean_peak_toll": float(lg["toll"][peak].mean()),
        "max_toll": float(lg["toll"].max()),
        "toll_changes": int((np.abs(np.diff(lg["toll"])) > 1e-9).sum()),
        "weave_vph_peak": float(lg["weave"][peak].mean() * 3600.0),
        "corridor_tt_gp_min": float(lg["tt_gp"][peak].mean() / 60.0),
    }


def score(m: dict[str, float]) -> float:
    """Selection score: peak corridor throughput, heavily penalised below 90% HOT reliability."""
    return m["hot_throughput_vph"] + m["gp_throughput_vph"] - 20_000.0 * max(0.0, 0.9 - m["hot_reliability_peak"])


def tune_pid(p: CorridorParams, seeds: list[int]) -> tuple[PID, list[dict[str, float]]]:
    """Grid-search PID gains and target on training seeds (the RL agent gets training seeds too)."""
    trials = []
    best: tuple[float, PID] | None = None
    for kp in (2.0, 4.0, 8.0):
        for ki in (0.2, 0.5, 1.0):
            for tgt in (0.8, 0.9, 1.0):
                ctrl = PID(p, target_frac=tgt, kp=kp, ki=ki, kd=1.0)
                sc = float(np.mean([score(metrics(episode(p, ctrl, s))) for s in seeds]))
                trials.append({"kp": kp, "ki": ki, "target_frac": tgt, "score": sc})
                if best is None or sc > best[0]:
                    best = (sc, ctrl)
    assert best is not None
    return best[1], trials


def drift_experiment(p: CorridorParams, seeds: list[int], shift: float = 0.6, at_s: float = 2.0 * 3600) -> dict[str, Any]:
    """Run PID-controlled days with and without a VOT shift; measure Page-Hinkley detection."""
    out: dict[str, Any] = {"shift_factor": shift, "shift_at_h": at_s / 3600.0, "detected": 0, "delays_min": [],
                           "false_alarms": 0, "seeds": len(seeds)}
    for s in seeds:
        for shifted in (False, True):
            c = episode(p, PID(p), s, vot_shift_at_s=at_s if shifted else None, vot_shift=shift)
            lg = c.state.log
            ph = PageHinkley()
            hit = None
            for t, obs_share, pred in zip(lg["t"][::18], lg["hot_share"][::18], lg["predicted_share"][::18]):
                if t < 1800.0:
                    continue
                if ph.update(obs_share - pred) and hit is None:
                    hit = t
            if shifted and hit is not None and hit >= at_s:
                out["detected"] += 1
                out["delays_min"].append((hit - at_s) / 60.0)
            if (not shifted and hit is not None) or (shifted and hit is not None and hit < at_s):
                out["false_alarms"] += 1
    out["median_delay_min"] = float(np.median(out["delays_min"])) if out["delays_min"] else math.nan
    return out


def run(out_dir: str | Path, *, train_episodes: int = 300, test_seeds: list[int] | None = None) -> dict[str, Any]:
    p = CorridorParams()
    out = Path(out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    test_seeds = test_seeds or list(range(1001, 1011))
    rl = QLearning(p, seed=0)
    t0 = time.perf_counter()
    curve = []
    for ep in range(train_episodes):
        rl.epsilon = max(0.02, 0.3 * (1 - ep / train_episodes))
        c = episode(p, rl, seed=ep, learn=True)  # training seeds 0..N-1, disjoint from test seeds
        curve.append(metrics(c)["hot_reliability_peak"])
    train_s = time.perf_counter() - t0
    rl.training = False
    np.save(out / "q_table.npy", rl.q)
    pid, pid_trials = tune_pid(p, list(range(500, 505)))  # tuning seeds, disjoint from test seeds

    ctrls = {"time_of_day": TimeOfDay(p), "density_lookup": DensityLookup(p), "pid": pid, "q_learning": rl}
    rows, logs = [], {}
    for name, ctrl in ctrls.items():
        for s in test_seeds:
            c = episode(p, ctrl, s)
            rows.append({"controller": name, "seed": s, **metrics(c)})
            if s == test_seeds[0]:
                logs[name] = pd.DataFrame(c.state.log)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_seed.csv", index=False)
    summ: dict[str, Any] = {"train_episodes": train_episodes, "train_seconds": train_s, "test_seeds": test_seeds,
                            "controllers": {}, "learning_curve": curve,
                            "pid_selected": {"kp": pid.kp, "ki": pid.ki, "kd": pid.kd, "target_frac": pid.target_frac},
                            "pid_trials": pid_trials}
    df["score"] = [score(r) for r in df.to_dict("records")]
    base = df[df["controller"] == "time_of_day"].sort_values("seed")
    for name in ctrls:
        sub = df[df["controller"] == name].sort_values("seed")
        e = {}
        for m in [c for c in df.columns if c not in ("controller", "seed")]:
            x = sub[m].to_numpy(float)
            ci = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)))
            e[m] = {"mean": float(x.mean()), "ci": ci}
            if name != "time_of_day" and m in ("hot_reliability_peak", "gp_mean_speed_mph", "entry_queue_veh_h"):
                e[m]["p_vs_tod"] = float(stats.ttest_rel(base[m].to_numpy(float), x).pvalue)
        summ["controllers"][name] = e
    summ["drift"] = drift_experiment(p, test_seeds)
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=float), encoding="utf-8")
    _plot(logs, curve, p, out / "figures")
    return summ


def _plot(logs: dict[str, pd.DataFrame], curve: list[float], p: CorridorParams, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    colors = {"time_of_day": "#868e96", "density_lookup": "#f59f00", "pid": "#1c7ed6", "q_learning": "#2b8a3e"}
    for name, lg in logs.items():
        h = lg["t"] / 3600.0
        axes[0].plot(h, lg["toll"], color=colors[name], label=name)
        axes[1].plot(h, lg["hot_speed"] / MPH, color=colors[name], label=name)
    axes[1].axhline(45, color="k", ls=":", lw=1)
    axes[0].set_ylabel("toll ($)")
    axes[1].set_ylabel("HOT minimum speed (mph)")
    for ax in axes[:2]:
        ax.set_xlabel("hour")
        ax.axvspan(1, 3, color="#f1f3f5", zorder=0)
    axes[0].legend(fontsize=8)
    w = 10
    sm = np.convolve(curve, np.ones(w) / w, mode="valid")
    axes[2].plot(np.arange(len(sm)) + w, 100 * sm, color="#2b8a3e")
    axes[2].set_xlabel("training episode")
    axes[2].set_ylabel("peak HOT reliability (%), 10-ep mean")
    fig.tight_layout()
    fig.savefig(fig_dir / "controllers.png", dpi=130)
    plt.close(fig)
