from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hotlane.api.app import create_app
from hotlane.controllers import PID, DensityLookup, QLearning, TimeOfDay
from hotlane.drift import PageHinkley
from hotlane.model import MPH, Corridor, CorridorParams
from hotlane.study import episode, metrics, score

P = CorridorParams(duration_s=2 * 3600.0)


def test_fundamental_diagram_constants() -> None:
    p = CorridorParams()
    assert p.q_max * 3600 == pytest.approx(1967, abs=2)
    assert p.vf * p.dt_s <= p.cell_m  # CFL


def test_conservation_without_congestion() -> None:
    p = CorridorParams(duration_s=3600.0, base_demand_vph=2000.0, peak_demand_vph=2000.0, demand_noise=0.0)
    c = Corridor(p, 1)
    for _ in range(int(p.duration_s / p.dt_s)):
        c.step(2.0)
    inside = c.state.k_gp.sum() * p.cell_m + c.state.k_hot.sum() * p.cell_m + c.state.queue
    out = (np.sum(c.state.log["flow_gp"]) + np.sum(c.state.log["flow_hot"])) * p.dt_s
    assert out + inside == pytest.approx(c.demand.sum() * p.dt_s, rel=1e-6)


def test_hot_share_falls_with_toll() -> None:
    c = Corridor(P, 1)
    shares = [c.hot_share(t, tt_gp=600.0, tt_hot=300.0) for t in (0.5, 2.0, 5.0, 10.0)]
    assert all(a > b for a, b in zip(shares, shares[1:]))
    assert c.hot_share(5.0, 300.0, 300.0) < 0.2


def test_low_flat_toll_breaks_down_the_hot_lane() -> None:
    cheap = metrics(episode(CorridorParams(), TimeOfDay(CorridorParams(), peak=1.0), 7))
    pricier = metrics(episode(CorridorParams(), TimeOfDay(CorridorParams(), peak=4.0), 7))
    assert cheap["hot_reliability_peak"] < 0.5 < pricier["hot_reliability_peak"]
    assert score(pricier) > score(cheap)


def test_controllers_respect_bounds() -> None:
    for ctrl in (TimeOfDay(P), DensityLookup(P), PID(P), QLearning(P)):
        c = episode(P, ctrl, 3, learn=isinstance(ctrl, QLearning))
        tolls = np.array(c.state.log["toll"])
        assert tolls.min() >= P.toll_min - 1e-9 and tolls.max() <= P.toll_max + 1e-9


def test_pid_raises_toll_when_dense() -> None:
    pid = PID(P)
    low = pid.toll({"k_hot": 0.5 * P.k_crit, "k_gp": 0.0, "toll": 1.0, "v_hot": 29.0, "t": 0.0})
    pid.reset()
    high = pid.toll({"k_hot": 1.2 * P.k_crit, "k_gp": 0.0, "toll": 1.0, "v_hot": 29.0, "t": 0.0})
    assert high > low


def test_q_learning_updates_values() -> None:
    q = QLearning(P, seed=1)
    episode(P, q, 5, learn=True)
    assert np.abs(q.q).sum() > 0
    assert QLearning.reward({"v_hot": 10.0}, 6000.0, 0.0) < QLearning.reward({"v_hot": 25.0}, 6000.0, 0.0)


def test_page_hinkley() -> None:
    rng = np.random.default_rng(0)
    ph = PageHinkley()
    assert not any(ph.update(x) for x in rng.normal(0, 0.01, 200))
    ph2 = PageHinkley()
    seq = np.r_[rng.normal(0, 0.01, 100), rng.normal(-0.1, 0.01, 100)]
    hits = [i for i, x in enumerate(seq) if ph2.update(x)]
    assert hits and 100 <= hits[0] < 130


def test_api() -> None:
    c = TestClient(create_app())
    r = c.post("/v1/toll", json={"controller": "density_lookup", "k_hot_veh_per_km": 20.0, "k_gp_veh_per_km": 25.0, "hour": 2.0})
    assert r.status_code == 200 and P.toll_min <= r.json()["toll_usd"] <= P.toll_max
    assert c.post("/v1/toll", json={"controller": "nope", "k_hot_veh_per_km": 1, "k_gp_veh_per_km": 1, "hour": 1}).status_code == 422
    assert MPH > 0
