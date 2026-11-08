"""Pricing controllers. All see the same observation every ``update_s`` and return a toll.

* :class:`TimeOfDay`   fixed schedule (shoulder / peak), the static baseline.
* :class:`DensityLookup` MnPASS-style table: the toll steps with HOT density, limited to one
  bin change per update.
* :class:`PID`         feedback on the HOT density error relative to a target below critical
  density, with output clamping and conditional-integration anti-windup.
* :class:`QLearning`   tabular Q-learning over (HOT density bin, GP density bin, HOT density trend,
  current toll bin) with toll increments as actions. Trained on training seeds, frozen (greedy) for
  evaluation. The toll bin matters: with incremental actions, the same action means different things
  at $1 and at $9, and without it the problem is partially observed (the first version omitted it and
  reached only 50% reliability).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .model import MPH, CorridorParams


class Controller(Protocol):
    name: str

    def reset(self) -> None: ...

    def toll(self, obs: dict[str, float]) -> float: ...


@dataclass
class TimeOfDay:
    p: CorridorParams
    shoulder: float = 1.0
    peak: float = 4.0
    name: str = "time_of_day"

    def reset(self) -> None:
        return None

    def toll(self, obs: dict[str, float]) -> float:
        h = obs["t"] / 3600.0
        return self.peak if 1.0 <= h < 3.0 else self.shoulder


@dataclass
class DensityLookup:
    p: CorridorParams
    edges_frac: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 1.05)  # of critical density
    tolls: tuple[float, ...] = (0.5, 1.0, 2.0, 3.5, 5.5, 8.0)
    name: str = "density_lookup"
    _bin: int = 0

    def reset(self) -> None:
        self._bin = 0

    def toll(self, obs: dict[str, float]) -> float:
        target = int(np.searchsorted(np.array(self.edges_frac) * self.p.k_crit, obs["k_hot"]))
        self._bin += int(np.sign(target - self._bin))  # at most one step per update
        return self.tolls[self._bin]


@dataclass
class PID:
    p: CorridorParams
    target_frac: float = 0.85  # of critical density
    kp: float = 6.0
    ki: float = 0.8
    kd: float = 2.0
    name: str = "pid"
    _e1: float = 0.0
    _e2: float = 0.0
    _u: float = 1.0

    def reset(self) -> None:
        self._e1, self._e2, self._u = 0.0, 0.0, 1.0

    def toll(self, obs: dict[str, float]) -> float:
        """Velocity-form PID: du = Kp (e - e1) + Ki e + Kd (e - 2 e1 + e2).

        Clamping the output of the velocity form is itself an anti-windup scheme: no integral
        state accumulates while the toll sits at a bound.
        """
        e = (obs["k_hot"] - self.target_frac * self.p.k_crit) / self.p.k_crit
        du = self.kp * (e - self._e1) + self.ki * e + self.kd * (e - 2 * self._e1 + self._e2)
        u = float(np.clip(self._u + du, self.p.toll_min, self.p.toll_max))
        self._e2, self._e1, self._u = self._e1, e, u
        return u


ACTIONS = np.array([-1.0, -0.25, 0.0, 0.25, 1.0])


@dataclass
class QLearning:
    p: CorridorParams
    alpha: float = 0.15
    gamma: float = 0.9
    epsilon: float = 0.1
    seed: int = 0
    name: str = "q_learning"
    q: np.ndarray = field(default_factory=lambda: np.zeros((7, 4, 3, 6, ACTIONS.size)))
    training: bool = True
    _rng: np.random.Generator = field(init=False)
    _prev: tuple[tuple[int, int, int, int], int] | None = None
    _last_k: float = 0.0
    _u: float = 1.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def reset(self) -> None:
        self._prev, self._last_k, self._u = None, 0.0, 1.0

    def _state(self, obs: dict[str, float]) -> tuple[int, int, int, int]:
        kc = self.p.k_crit
        h = int(np.searchsorted(np.array([0.5, 0.7, 0.85, 0.95, 1.05, 1.3]) * kc, obs["k_hot"]))
        g = int(np.searchsorted(np.array([0.6, 1.0, 1.6]) * kc, obs["k_gp"]))
        dk = obs["k_hot"] - self._last_k
        tr = 0 if dk < -0.02 * kc else (2 if dk > 0.02 * kc else 1)
        tb = int(np.searchsorted(np.array([1.0, 2.0, 3.5, 5.0, 7.5]), self._u))
        return h, g, tr, tb

    @staticmethod
    def reward(obs: dict[str, float], corridor_flow_vph: float, toll_change: float) -> float:
        """Total corridor throughput, a hard penalty below 45 mph (federal managed-lane standard), toll churn.

        The breakdown penalty (20, in thousands of veh/h) matches the evaluation score's reliability
        constraint. With a penalty of 5 the agent learned to under-price and trade HOT breakdowns for
        throughput, which is exactly what that reward asked for.
        """
        breakdown = obs["v_hot"] < 45.0 * MPH
        return corridor_flow_vph / 1000.0 - 20.0 * float(breakdown) - 0.2 * abs(toll_change)

    def learn(self, obs: dict[str, float], r: float) -> None:
        if not self.training or self._prev is None:
            return
        s, a = self._prev
        s2 = self._state(obs)
        td = r + self.gamma * self.q[s2].max() - self.q[s + (a,)]
        self.q[s + (a,)] += self.alpha * td

    def toll(self, obs: dict[str, float]) -> float:
        s = self._state(obs)
        if self.training and self._rng.random() < self.epsilon:
            a = int(self._rng.integers(ACTIONS.size))
        else:
            a = int(np.argmax(self.q[s]))
        self._prev = (s, a)
        self._last_k = obs["k_hot"]
        self._u = float(np.clip(self._u + ACTIONS[a], self.p.toll_min, self.p.toll_max))
        return self._u
