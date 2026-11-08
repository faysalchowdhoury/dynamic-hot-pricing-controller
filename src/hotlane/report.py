"""Markdown report."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

LAB = {"time_of_day": "Time-of-day ($1 / $4)", "density_lookup": "Density lookup (MnPASS-style)",
       "pid": "PID (tuned)", "q_learning": "Q-learning (trained)"}
ROWS = [("hot_reliability_peak", "Peak HOT reliability (min speed >= 45 mph)", "{:.0%}", 1.0),
        ("hot_throughput_vph", "HOT throughput, peak (veh/h)", "{:.0f}", 1.0),
        ("gp_throughput_vph", "GP throughput, peak (veh/h)", "{:.0f}", 1.0),
        ("gp_mean_speed_mph", "GP mean speed, peak (mph)", "{:.1f}", 1.0),
        ("entry_queue_veh_h", "Upstream queue (veh-h)", "{:.0f}", 1.0),
        ("mean_peak_toll", "Mean peak toll ($)", "{:.2f}", 1.0),
        ("max_toll", "Max toll ($)", "{:.2f}", 1.0),
        ("toll_changes", "Toll changes per day", "{:.0f}", 1.0),
        ("revenue_usd", "Revenue per 4-h period ($)", "{:,.0f}", 1.0),
        ("weave_vph_peak", "Ingress weaving flow, peak (veh/h)", "{:.0f}", 1.0)]


def write_report(s: dict[str, Any], out: Path) -> Path:
    names = list(LAB)
    L = ["# HOT-lane pricing study", "",
         f"Q-learning trained for {s['train_episodes']} episodes on seeds 0-{s['train_episodes'] - 1} "
         f"({s['train_seconds']:.0f} s); PID gains grid-searched on seeds 500-504 "
         f"(selected Kp={s['pid_selected']['kp']}, Ki={s['pid_selected']['ki']}, target={s['pid_selected']['target_frac']} x k_crit). "
         f"All controllers evaluated on held-out seeds {s['test_seeds'][0]}-{s['test_seeds'][-1]} (mean ± 95% CI).", "",
         "| Metric | " + " | ".join(LAB[n] for n in names) + " |", "|" + "---|" * (len(names) + 1)]
    for key, label, fmt, _ in ROWS:
        cells = []
        for n in names:
            m = s["controllers"][n][key]
            if "%" in fmt:
                cell = f"{100 * m['mean']:.0f}% ± {100 * m['ci']:.0f}"
            else:
                cell = fmt.format(m["mean"]) + " ± " + fmt.format(m["ci"])
            cells.append(cell)
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    d = s["drift"]
    L += ["", "## Drift detection (Page-Hinkley on observed - predicted HOT share, PID-controlled days)", "",
          f"VOT multiplied by {d['shift_factor']} at hour {d['shift_at_h']:.1f}: detected on {d['detected']}/{d['seeds']} days, "
          f"median delay {d['median_delay_min']:.0f} min; false alarms {d['false_alarms']} over {2 * d['seeds']} days "
          "(shifted days alarming before the shift + unshifted days).", "", "![Controllers](figures/controllers.png)", ""]
    md = out / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
