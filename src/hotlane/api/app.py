"""``POST /v1/toll``: stateless toll recommendation from current densities (lookup / PID step / Q-policy)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .. import __version__
from ..controllers import PID, DensityLookup, QLearning, TimeOfDay
from ..model import CorridorParams


class TollRequest(BaseModel):
    controller: Literal["time_of_day", "density_lookup", "pid", "q_learning"]
    k_hot_veh_per_km: float = Field(ge=0, le=200)
    k_gp_veh_per_km: float = Field(ge=0, le=200)
    hour: float = Field(ge=0, le=24)
    current_toll: float = Field(1.0, ge=0)


def create_app(q_table: Path | None = None) -> FastAPI:
    p = CorridorParams()
    app = FastAPI(title="HOT lane pricing", version=__version__)
    q = QLearning(p, training=False)
    if q_table is not None and q_table.is_file():
        q.q = np.load(q_table)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/toll")
    def toll(req: TollRequest) -> dict[str, float]:
        obs = {"k_hot": req.k_hot_veh_per_km / 1000.0, "k_gp": req.k_gp_veh_per_km / 1000.0, "toll": req.current_toll,
               "v_hot": p.vf, "t": req.hour * 3600.0}
        ctrl = {"time_of_day": TimeOfDay(p), "density_lookup": DensityLookup(p), "pid": PID(p), "q_learning": q}[req.controller]
        if isinstance(ctrl, (PID, QLearning)):
            ctrl._u = req.current_toll  # continue from the toll currently posted
        if isinstance(ctrl, DensityLookup):
            ctrl._bin = 5  # stateless call: jump straight to the target bin
            target = int(np.searchsorted(np.array(ctrl.edges_frac) * p.k_crit, obs["k_hot"]))
            return {"toll_usd": float(ctrl.tolls[target])}
        return {"toll_usd": float(np.clip(ctrl.toll(obs), p.toll_min, p.toll_max))}

    return app
