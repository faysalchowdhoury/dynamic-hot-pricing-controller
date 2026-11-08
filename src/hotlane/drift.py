"""Page-Hinkley change detection on the price-response model.

The operator's lane-choice model predicts the SOV share that will buy in at the current toll and
travel times. If the population's willingness to pay shifts (fuel prices, recession, remote work), the
observed share departs from the prediction. Page-Hinkley flags a sustained shift in the mean of the
residual (observed - predicted), in either direction. That is the signal to re-estimate the
value-of-time distribution and retune the controller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageHinkley:
    delta: float = 0.005  # tolerated drift magnitude
    threshold: float = 0.6
    burn_in: int = 30
    n: int = 0
    mean: float = 0.0
    m_up: float = 0.0
    m_down: float = 0.0
    min_up: float = 0.0
    max_down: float = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.m_up += x - self.mean - self.delta
        self.m_down += x - self.mean + self.delta
        self.min_up = min(self.min_up, self.m_up)
        self.max_down = max(self.max_down, self.m_down)
        if self.n < self.burn_in:
            return False
        return (self.m_up - self.min_up) > self.threshold or (self.max_down - self.m_down) > self.threshold
