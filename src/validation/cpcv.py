"""
Combinatorial Purged Cross-Validation (CPCV).
Implements Marcos Lopez de Prado's leakage-proof validation framework with purging and embargoing.
"""

from typing import List, Tuple, Generator, Dict
import itertools
import math
import numpy as np
import pandas as pd


class CombinatorialPurgedCV:
    """
    Constructs all binom(N, k) train-test combinations across contiguous time blocks.
    Enforces forward label purging and post-test embargoing.
    """

    def __init__(
        self,
        n_blocks: int = 6,
        k_test_blocks: int = 2,
        embargo_pct: float = 0.02,
    ):
        self.n_blocks = n_blocks
        self.k_test_blocks = k_test_blocks
        self.embargo_pct = embargo_pct

    def split(
        self,
        df: pd.DataFrame,
        label_holding_bars: int = 20,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yields (train_indices, test_indices) for each combination.
        
        Args:
            df: Feature/Target DataFrame with sorted DatetimeIndex.
            label_holding_bars: Maximum forward barrier lookahead h.
        """
        n_samples = len(df)
        block_size = n_samples // self.n_blocks
        blocks = []

        # Partition index into N contiguous blocks
        for i in range(self.n_blocks):
            start_idx = i * block_size
            end_idx = (i + 1) * block_size if i < self.n_blocks - 1 else n_samples
            blocks.append((start_idx, end_idx))

        # Generate all binom(N, k) combinations of test blocks
        test_combos = list(itertools.combinations(range(self.n_blocks), self.k_test_blocks))
        embargo_bars = int(self.embargo_pct * block_size)

        for combo in test_combos:
            test_mask = np.zeros(n_samples, dtype=bool)
            test_intervals = []

            for b_idx in combo:
                b_start, b_end = blocks[b_idx]
                test_mask[b_start:b_end] = True
                test_intervals.append((b_start, b_end))

            train_mask = ~test_mask

            # Apply Purging: remove training bars whose forward label intersects any test interval
            # If t + h >= t_test_start, purge t
            for b_start, b_end in test_intervals:
                purge_start = max(0, b_start - label_holding_bars)
                train_mask[purge_start:b_start] = False

                # Apply Embargoing: remove post-test observations to prevent AR leak
                embargo_end = min(n_samples, b_end + embargo_bars)
                train_mask[b_end:embargo_end] = False

            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]

            yield train_indices, test_indices

    def num_paths(self) -> int:
        """Total distinct out-of-sample backtest paths: binom(N, k) * (k / N)."""
        combos = math.comb(self.n_blocks, self.k_test_blocks)
        return int(combos * (self.k_test_blocks / self.n_blocks))
