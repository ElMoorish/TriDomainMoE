"""
Multi-Resolution Concept Drift Detection (MRDD) Engine.
Implements Page 12 of Whitepaper:
  1. Discrete Wavelet Transform (DWT) multi-scale energy divergence
  2. Page-Hinkley non-parametric cumulative monitor on Information Coefficient (IC) decay
"""

from typing import Dict, Any, List, Tuple
import numpy as np


class WaveletFilterBank:
    """Native numpy 1D Discrete Wavelet Transform (Haar & D4 wavelets)."""

    @staticmethod
    def dwt_decomposition(signal: np.ndarray, levels: int = 3) -> List[np.ndarray]:
        """
        Decomposes 1D signal into J detail coefficient scales: [d_1, d_2, ..., d_J, a_J].
        Uses Haar wavelet filter bank for maximal execution speed.
        """
        coeffs = []
        current_approx = signal.copy()
        
        # Ensure even length
        for j in range(levels):
            if len(current_approx) < 4:
                break
            if len(current_approx) % 2 != 0:
                current_approx = current_approx[:-1]

            # Haar filter: approx = (s[2k] + s[2k+1]) / sqrt(2), detail = (s[2k] - s[2k+1]) / sqrt(2)
            even = current_approx[0::2]
            odd = current_approx[1::2]
            approx = (even + odd) / np.sqrt(2.0)
            detail = (even - odd) / np.sqrt(2.0)

            coeffs.append(detail)
            current_approx = approx

        return coeffs


class MultiResolutionDriftDetector:
    """
    Monitors distribution shifts across multiple time horizons simultaneously.
    Combines Wavelet Energy differences with Page-Hinkley cumulative IC tracking.
    """

    def __init__(
        self,
        wavelet_levels: int = 3,
        ph_delta: float = 0.005,
        ph_threshold: float = 0.05,
        energy_threshold: float = 0.25,
    ):
        self.levels = wavelet_levels
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.energy_threshold = energy_threshold

        # Page-Hinkley state variables
        self._u_t: float = 0.0
        self._m_t: float = 0.0
        self._ic_history: List[float] = []

        # Reference wavelet energy baseline across J scales
        self._ref_energies: Dict[int, float] = {}

    def set_reference_baseline(self, ref_returns: np.ndarray) -> None:
        """Establish baseline wavelet energy distribution from historical in-sample data."""
        details = WaveletFilterBank.dwt_decomposition(ref_returns, levels=self.levels)
        for j, dj in enumerate(details):
            scale = j + 1
            # Energy = (1 / 2^j) * sum(|d_{j,k}|^2)
            energy = float(np.sum(dj ** 2) / (2.0 ** scale))
            self._ref_energies[scale] = max(energy, 1e-8)

    def evaluate_wavelet_drift(self, test_returns: np.ndarray) -> Dict[str, Any]:
        """
        D_{wavelet}^{(j)} = (1 / 2^j) * sum(|d_{j,k}^test|^2) - (1 / 2^j) * sum(|d_{j,k}^ref|^2)
        """
        if not self._ref_energies:
            self.set_reference_baseline(test_returns)

        details = WaveletFilterBank.dwt_decomposition(test_returns, levels=self.levels)
        energy_diffs = {}
        max_divergence = 0.0

        for j, dj in enumerate(details):
            scale = j + 1
            test_energy = float(np.sum(dj ** 2) / (2.0 ** scale))
            ref_energy = self._ref_energies.get(scale, 1e-6)

            # Relative energy divergence
            rel_diff = abs(test_energy - ref_energy) / ref_energy
            energy_diffs[f"scale_{scale}"] = rel_diff
            if rel_diff > max_divergence:
                max_divergence = rel_diff

        drift_detected = bool(max_divergence >= self.energy_threshold)
        return {
            "wavelet_drift_detected": drift_detected,
            "max_divergence": max_divergence,
            "energy_diffs_by_scale": energy_diffs,
        }

    def update_page_hinkley(self, current_ic: float) -> Tuple[bool, float]:
        """
        Page-Hinkley downward IC decay monitor:
        U_t = sum_{i=1}^t (IC_bar - IC_i - delta_m)
        m_t = min_{1 <= i <= t} U_i
        Alert triggered when U_t - m_t >= lambda_threshold.
        """
        self._ic_history.append(current_ic)
        ic_bar = float(np.mean(self._ic_history))

        # Divergence accumulates when current IC drops significantly below historical average
        delta_step = (ic_bar - current_ic - self.ph_delta)
        self._u_t += delta_step
        if len(self._ic_history) == 1 or self._u_t < self._m_t:
            self._m_t = self._u_t

        divergence = max(0.0, self._u_t - self._m_t)
        alarm = bool(divergence >= self.ph_threshold)

        if alarm:
            # Reset Page-Hinkley monitor upon triggering
            self._u_t = 0.0
            self._m_t = 0.0

        return alarm, float(divergence)
