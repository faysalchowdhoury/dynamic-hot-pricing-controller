"""Freeway corridor with a single-lane HOT facility.

Two parallel cell transmission models (CTM, triangular fundamental diagram) share one entry:

* **GP**: 3 lanes with a recurring bottleneck (lane drop / merge) at 85% of capacity near
  the downstream end;
* **HOT**: 1 lane, buffer-separated, single ingress just downstream of the entry.

**Lane choice** happens at the ingress every step. HOVs always use the HOT lane. Each SOV
compares ``U_HOT = -toll - VOT * TT_HOT + beta`` with ``U_GP = -VOT * TT_GP`` (logit, scale
$1), where travel times are the *current* corridor travel times (what a dynamic message sign
would show) and VOT is lognormal, median $18/h. The HOT share is the mean choice probability
over 512 VOT draws.

**Weaving turbulence**: vehicles crossing from the GP lanes into the ingress weave reduce GP
capacity in the ingress cell by ``alpha * f_LC / q_max`` (alpha = 0.6): the more drivers the
toll pulls across, the more the GP lanes lose at the access point.

Units: metres, seconds, vehicles; tolls in dollars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

MPH = 0.44704


class CorridorParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    length_m: float = 8000.0
    cell_m: float = 400.0
    dt_s: float = 10.0
    gp_lanes: int = 3
    hot_lanes: int = 1
    vf: float = 29.0
    w: float = 5.4
    kj: float = 0.12
    bottleneck_cell: int = 16
    bottleneck_factor: float = 0.85
    hot_egress_factor: float = 0.90  # HOT egress merge back into GP
    ingress_cell: int = 2
    weave_alpha: float = 0.6
    hov_share: float = 0.12
    vot_median: float = 18.0  # $/h
    vot_sigma: float = 0.6
    hot_asc: float = 0.3  # $ equivalent of reliability / comfort of the managed lane
    logit_scale: float = 1.0
    toll_min: float = 0.5
    toll_max: float = 10.0
    duration_s: float = 4 * 3600.0
    base_demand_vph: float = 4300.0
    peak_demand_vph: float = 7000.0
    demand_noise: float = 0.06

    @property
    def n_cells(self) -> int:
        return int(round(self.length_m / self.cell_m))

    @property
    def q_max(self) -> float:
        return self.vf * self.w * self.kj / (self.vf + self.w)

    @property
    def k_crit(self) -> float:
        return self.q_max / self.vf


def demand_profile(p: CorridorParams, rng: np.random.Generator) -> np.ndarray:
    """Entering veh/s per step: shoulder -> peak (trapezoid) -> shoulder, AR(1) noise."""
    n = int(p.duration_s / p.dt_s)
    t = np.arange(n) * p.dt_s / 3600.0
    shape = np.interp(t, [0.0, 0.75, 1.5, 2.5, 3.25, 4.0], [0.0, 0.2, 1.0, 1.0, 0.2, 0.0])
    base = p.base_demand_vph + (p.peak_demand_vph - p.base_demand_vph) * shape
    eps = np.zeros(n)
    for i in range(1, n):
        eps[i] = 0.97 * eps[i - 1] + rng.normal(0, p.demand_noise * 0.25)
    return np.maximum(base * (1 + eps), 0.0) / 3600.0


@dataclass
class State:
    k_gp: np.ndarray
    k_hot: np.ndarray
    queue: float = 0.0
    toll: float = 1.0
    t: float = 0.0
    log: dict[str, list[float]] = field(default_factory=lambda: {k: [] for k in (
        "t", "toll", "hot_share", "hot_density", "hot_speed", "gp_speed", "gp_density", "tt_gp", "tt_hot",
        "flow_hot", "flow_gp", "weave", "queue", "revenue", "predicted_share")})


class Corridor:
    def __init__(self, p: CorridorParams, seed: int, *, vot_shift_at_s: float | None = None, vot_shift: float = 1.0) -> None:
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.demand = demand_profile(p, self.rng)
        self.vot = self.rng.lognormal(np.log(p.vot_median), p.vot_sigma, 512)
        self.vot_shift_at_s = vot_shift_at_s
        self.vot_shift = vot_shift
        n = p.n_cells
        self.state = State(np.zeros(n), np.zeros(n))
        # upstream of the ingress the managed lane's footprint is part of the general cross-section
        self.lanes_gp = np.where(np.arange(n) <= p.ingress_cell, p.gp_lanes + p.hot_lanes, p.gp_lanes).astype(float)
        self.cap_gp = p.q_max * self.lanes_gp
        self.cap_gp[p.bottleneck_cell] *= p.bottleneck_factor
        self.cap_hot = np.full(n, p.q_max * p.hot_lanes)
        self.cap_hot[-1] *= p.hot_egress_factor

    # ---------------------------------------------------------------- choice
    def travel_times(self) -> tuple[float, float]:
        p, s = self.p, self.state
        v_gp = np.clip(self._speed(s.k_gp, self.lanes_gp), 1.0, p.vf)
        v_hot = np.clip(self._speed(s.k_hot, p.hot_lanes), 1.0, p.vf)
        return float(np.sum(p.cell_m / v_gp)), float(np.sum(p.cell_m / v_hot))

    def _speed(self, k: np.ndarray, lanes: float | np.ndarray) -> np.ndarray:
        p = self.p
        kl = k / lanes
        q = np.minimum(p.vf * kl, p.w * (p.kj - kl))
        return np.where(kl > 1e-6, q / np.maximum(kl, 1e-6), p.vf)

    def hot_share(self, toll: float, tt_gp: float, tt_hot: float, vot: np.ndarray | None = None) -> float:
        p = self.p
        v = self.vot if vot is None else vot
        dv = (-toll - v * tt_hot / 3600.0 + p.hot_asc) - (-v * tt_gp / 3600.0)
        return float(np.mean(1.0 / (1.0 + np.exp(-dv / p.logit_scale))))

    # ------------------------------------------------------------------ step
    def step(self, toll: float) -> dict[str, float]:
        p, s = self.p, self.state
        toll = float(np.clip(toll, p.toll_min, p.toll_max))
        i = int(s.t / p.dt_s)
        tt_gp, tt_hot = self.travel_times()
        vot = self.vot * (self.vot_shift if (self.vot_shift_at_s is not None and s.t >= self.vot_shift_at_s) else 1.0)
        share_sov = self.hot_share(toll, tt_gp, tt_hot, vot)
        predicted = self.hot_share(toll, tt_gp, tt_hot)  # nominal model, used for drift monitoring
        s.queue += self.demand[i] * p.dt_s
        dt, dx = p.dt_s, p.cell_m

        def flows(k: np.ndarray, cap: np.ndarray, lanes: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            send = np.minimum(p.vf * k, cap)
            recv = np.minimum(cap, np.maximum(p.w * (p.kj * lanes - k), 0.0))
            return send, recv

        # entry: everybody enters the shared upstream section; diversion to HOT happens at the ingress cell
        send_g, recv_g = flows(s.k_gp, self.cap_gp, self.lanes_gp)
        q_in = min(s.queue / dt, recv_g[0])
        s.queue -= q_in * dt
        g = p.ingress_cell
        hot_frac = p.hov_share + (1 - p.hov_share) * share_sov
        divert_want = send_g[g] * hot_frac
        send_h, recv_h = flows(s.k_hot, self.cap_hot, p.hot_lanes)
        divert = min(divert_want, recv_h[0])
        weave = divert
        cap_eff = self.cap_gp.copy()
        cap_eff[g] = self.cap_gp[g] * max(0.5, 1.0 - p.weave_alpha * weave / (p.q_max * p.gp_lanes))
        send_g, recv_g = flows(s.k_gp, cap_eff, self.lanes_gp)
        inner_g = np.minimum(send_g[:-1], recv_g[1:])
        # the ingress cell sends (send - divert) downstream in GP and divert into HOT cell 0
        inner_g[g] = min(max(send_g[g] - divert, 0.0), recv_g[g + 1])
        out_g = send_g[-1]
        inner_h = np.minimum(send_h[:-1], recv_h[1:])
        out_h = send_h[-1]
        kg, kh = s.k_gp, s.k_hot
        kg[0] += (q_in - inner_g[0]) * dt / dx
        kg[1:-1] += (inner_g[:-1] - inner_g[1:]) * dt / dx
        kg[-1] += (inner_g[-1] - out_g) * dt / dx
        kg[g] -= divert * dt / dx
        kh[0] += (divert - inner_h[0]) * dt / dx
        kh[1:-1] += (inner_h[:-1] - inner_h[1:]) * dt / dx
        kh[-1] += (inner_h[-1] - out_h) * dt / dx
        np.clip(kg, 0, p.kj * self.lanes_gp, out=kg)
        np.clip(kh, 0, p.kj * p.hot_lanes, out=kh)
        s.t += dt
        s.toll = toll
        v_hot = float(np.min(self._speed(kh, p.hot_lanes)))
        v_gp = float(np.mean(self._speed(kg, self.lanes_gp)))
        rec = {"t": s.t, "toll": toll, "hot_share": share_sov, "hot_density": float(kh.max() / p.hot_lanes),
               "hot_speed": v_hot, "gp_speed": v_gp, "gp_density": float(np.mean(kg / self.lanes_gp)), "tt_gp": tt_gp,
               "tt_hot": tt_hot, "flow_hot": out_h, "flow_gp": out_g, "weave": weave, "queue": s.queue,
               # only SOVs pay: they are (1 - hov_share / hot_frac) of the vehicles entering the HOT lane
               "revenue": divert * dt * toll * max(0.0, 1.0 - p.hov_share / max(hot_frac, 1e-9)),
               "predicted_share": predicted}
        for k, v in rec.items():
            s.log[k].append(v)
        return rec

    def observe(self) -> dict[str, float]:
        """What a controller sees: HOT max density (veh/m/ln), GP mean density, current toll, HOT speed."""
        p, s = self.p, self.state
        return {"k_hot": float(s.k_hot.max() / p.hot_lanes), "k_gp": float(np.mean(s.k_gp / self.lanes_gp)), "toll": s.toll,
                "v_hot": float(np.min(self._speed(s.k_hot, p.hot_lanes))), "t": s.t}
