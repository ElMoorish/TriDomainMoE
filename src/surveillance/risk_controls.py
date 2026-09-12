"""
Deterministic Risk Controls: Volatility Targeting and 3-Tier Drawdown Circuit Breakers.
Implements GARCH/EWMA inverse volatility position sizing and 3% / 5% / 8% trailing drawdown limits.
"""

from typing import Dict, Any, Tuple
from enum import Enum
import numpy as np


class CircuitBreakerState(Enum):
    NORMAL = "NORMAL"
    LEVERAGE_CUT = "LEVERAGE_CUT_50"  # Triggered at 3% drawdown
    CASH_OUT = "CASH_OUT_100"        # Triggered at 5% drawdown
    HALT_RETRAIN = "SYSTEM_HALT"      # Triggered at 8% drawdown


class RiskControls:
    """
    Capital preservation engine enforcing dynamic volatility scaling
    and hard three-tier drawdown circuit breakers.
    """

    def __init__(
        self,
        target_vol: float = 0.15,
        max_leverage_cap: float = 2.0,
        dd_tier1: float = 0.03,  # 3% drawdown
        dd_tier2: float = 0.05,  # 5% drawdown
        dd_tier3: float = 0.08,  # 8% drawdown
    ):
        self.target_vol = target_vol
        self.max_cap = max_leverage_cap
        self.dd_tier1 = dd_tier1
        self.dd_tier2 = dd_tier2
        self.dd_tier3 = dd_tier3

        self.peak_equity: float = 1.0
        self.current_state: CircuitBreakerState = CircuitBreakerState.NORMAL

    @classmethod
    def strict_prop_profile(cls, max_dd_ceiling: float = 0.025) -> "RiskControls":
        """
        Creates an ultra-strict capital preservation profile for 2.5% max drawdown.
        Tier 1 (50% leverage cut): 1.25% DD
        Tier 2 (Cash liquidation): 2.00% DD
        Tier 3 (Execution shutdown): 2.50% DD
        """
        return cls(
            target_vol=0.12,
            max_leverage_cap=1.5,
            dd_tier1=max_dd_ceiling * 0.50,  # 1.25%
            dd_tier2=max_dd_ceiling * 0.80,  # 2.00%
            dd_tier3=max_dd_ceiling,         # 2.50%
        )

    def compute_vol_target_size(
        self,
        raw_direction: float,
        conviction_size: float,
        realized_vol: float,
    ) -> float:
        """
        Target Position = sign(y_hat) * min(Cap_max, sigma* / sigma_t) * s_t
        """
        vol_safe = max(1e-4, realized_vol)
        vol_scalar = min(self.max_cap, self.target_vol / vol_safe)
        sign = 1.0 if raw_direction >= 0 else -1.0
        return sign * vol_scalar * conviction_size

    def update_drawdown_monitor(self, current_equity: float) -> Tuple[CircuitBreakerState, float]:
        """
        Update trailing peak equity and check three-tier circuit breaker thresholds.
        Returns:
            Tuple of (CircuitBreakerState, current_drawdown_fraction)
        """
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        drawdown = (self.peak_equity - current_equity) / (self.peak_equity + 1e-8)

        if drawdown >= self.dd_tier3:
            self.current_state = CircuitBreakerState.HALT_RETRAIN
        elif drawdown >= self.dd_tier2:
            self.current_state = CircuitBreakerState.CASH_OUT
        elif drawdown >= self.dd_tier1:
            self.current_state = CircuitBreakerState.LEVERAGE_CUT
        else:
            self.current_state = CircuitBreakerState.NORMAL

        return self.current_state, float(drawdown)

    def apply_circuit_breaker(self, proposed_size: float) -> float:
        """Apply state-dependent position downscaling."""
        if self.current_state == CircuitBreakerState.HALT_RETRAIN:
            return 0.0
        elif self.current_state == CircuitBreakerState.CASH_OUT:
            return 0.0
        elif self.current_state == CircuitBreakerState.LEVERAGE_CUT:
            return proposed_size * 0.5
        return proposed_size
