"""
Experiment 3: Higher Brand Availability Probability (80%)

Change from baseline:
  - AVAIL_PROB: 0.60 → 0.80

Rationale:
  Higher availability means agents are more likely to find their preferred
  brands in stock. This is expected to:
  - Improve completion rates across all agents (less stockout risk)
  - Allow quality-focused agents (TimeAware, Quality-Threshold) to be
    more selective without sacrificing completion
  - Reduce the need for multi-store visits (preferred brand more likely
    at the first store visited)
  - Potentially reduce the reward gap between deep RL and heuristics
    (easier environment narrows the advantage of learned exploration)

This experiment tests how robust agent rankings are to availability changes,
and whether the learned discount-seeking behaviour of DQN/Double-DQN persists
when brands are more plentiful.
"""

import numpy as np
from typing import Optional
from environment import MarketEnvironment


class MarketEnvExp3(MarketEnvironment):
    """
    Experiment 3: Availability probability raised from 60% to 80%.
    All other parameters identical to baseline.
    """

    AVAIL_PROB = 0.80   # only change from baseline

    def __init__(self, seed: Optional[int] = None):
        super().__init__(seed=seed)
