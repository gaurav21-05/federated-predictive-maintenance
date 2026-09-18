"""
Privacy Accountant Module
Implements Rényi Differential Privacy (RDP) composition and moments accountant
for tracking cumulative (epsilon, delta) privacy expenditure across federated communication rounds.
"""

from typing import Dict, List, Optional, Tuple
import math
import numpy as np


class PrivacyAccountant:
    """
    Rényi Differential Privacy (RDP) Accountant for Gaussian Mechanism.
    
    Tracks cumulative privacy budget expenditure (epsilon, delta) over T federated rounds:
        R_T(alpha) = sum_{t=1}^T [ alpha / (2 * sigma_t^2) ]
        epsilon(delta) = min_{alpha > 1} [ R_T(alpha) + ln(1 / delta) / (alpha - 1) ]
    """

    DEFAULT_ORDERS = [
        1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0,
        32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 128.0, 256.0
    ]

    def __init__(
        self,
        target_delta: float = 1e-5,
        subsampling_ratio: float = 1.0,
        orders: Optional[List[float]] = None,
    ):
        """
        Args:
            target_delta: Allowed probability of privacy guarantee failure (typically 1e-5).
            subsampling_ratio: Subsampling ratio q in (0, 1]. For local model DP q=1.0. For mini-batch DP q=B/N.
            orders: List of Rényi orders alpha > 1 to evaluate for optimal epsilon conversion.
        """
        if target_delta <= 0 or target_delta >= 1:
            raise ValueError(f"target_delta must be in (0, 1), got {target_delta}")
        if subsampling_ratio <= 0 or subsampling_ratio > 1.0:
            raise ValueError(f"subsampling_ratio must be in (0, 1], got {subsampling_ratio}")

        self.target_delta = float(target_delta)
        self.subsampling_ratio = float(subsampling_ratio)
        self.orders = np.array(orders or self.DEFAULT_ORDERS, dtype=float)
        if np.any(self.orders <= 1.0):
            raise ValueError("All Rényi orders alpha must be strictly greater than 1.0")

        # Cumulative Rényi divergence for each order alpha
        self.rdp = np.zeros_like(self.orders)
        self.history: List[Dict[str, float]] = []
        self.total_rounds = 0

    def step(self, noise_multiplier: float) -> Tuple[float, float]:
        """
        Records completion of one communication round with the given noise multiplier sigma.
        
        Args:
            noise_multiplier: sigma ratio (noise_std / clip_norm). If sigma == 0 (no privacy),
                              returns epsilon = infinity.
                              
        Returns:
            (epsilon, delta): The updated cumulative privacy spent.
        """
        self.total_rounds += 1
        if noise_multiplier <= 0.0:
            current_eps = float("inf")
            self.history.append({
                "round": self.total_rounds,
                "noise_multiplier": 0.0,
                "epsilon": float("inf"),
                "delta": self.target_delta,
            })
            return float("inf"), self.target_delta

        # Add single-step RDP for Gaussian mechanism with optional subsampling amplification:
        # R(alpha) <= q^2 * alpha / (2 * sigma^2)
        step_rdp = (self.subsampling_ratio ** 2) * self.orders / (2.0 * (noise_multiplier ** 2))
        self.rdp += step_rdp

        current_eps, opt_alpha = self.get_epsilon(self.target_delta)
        self.history.append({
            "round": self.total_rounds,
            "noise_multiplier": float(noise_multiplier),
            "epsilon": float(current_eps),
            "delta": float(self.target_delta),
            "subsampling_ratio": float(self.subsampling_ratio),
            "optimal_order": float(opt_alpha),
        })
        return current_eps, self.target_delta

    def get_epsilon(self, delta: Optional[float] = None) -> Tuple[float, float]:
        """
        Converts cumulative RDP vector to (epsilon, delta)-DP by optimizing over Rényi orders:
        epsilon(delta) = min_{alpha > 1} [ R_T(alpha) + ln(1 / delta) / (alpha - 1) ]
        
        Returns:
            (min_epsilon, optimal_alpha)
        """
        if self.total_rounds == 0:
            return 0.0, float(self.orders[0])

        d = float(delta) if delta is not None else self.target_delta
        if d <= 0 or d >= 1:
            raise ValueError(f"delta must be in (0, 1), got {d}")

        eps_candidates = self.rdp + np.log(1.0 / d) / (self.orders - 1.0)
        min_idx = np.argmin(eps_candidates)
        return float(eps_candidates[min_idx]), float(self.orders[min_idx])

    @classmethod
    def compute_rdp_epsilon(
        cls,
        noise_multiplier: float,
        num_rounds: int,
        delta: float = 1e-5,
        subsampling_ratio: float = 1.0,
        orders: Optional[List[float]] = None,
    ) -> float:
        """
        Static analytical helper to calculate the total epsilon for a given configuration
        without instantiating full state.
        """
        if noise_multiplier <= 0.0:
            return float("inf")
        if num_rounds <= 0:
            return 0.0

        eval_orders = np.array(orders or cls.DEFAULT_ORDERS, dtype=float)
        cumulative_rdp = (num_rounds * (subsampling_ratio ** 2) * eval_orders) / (2.0 * (noise_multiplier ** 2))
        eps_candidates = cumulative_rdp + np.log(1.0 / delta) / (eval_orders - 1.0)
        return float(np.min(eps_candidates))

    @classmethod
    def calibrate_noise_multiplier(
        cls,
        target_epsilon: float,
        num_rounds: int,
        delta: float = 1e-5,
        subsampling_ratio: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 50,
    ) -> float:
        """
        Finds the exact noise multiplier sigma needed to achieve target_epsilon over num_rounds
        via binary search.
        """
        if target_epsilon <= 0:
            raise ValueError(f"target_epsilon must be positive, got {target_epsilon}")
        if target_epsilon == float("inf"):
            return 0.0

        low = 1e-3
        high = 50.0

        # Expand search range if needed
        while cls.compute_rdp_epsilon(high, num_rounds, delta, subsampling_ratio=subsampling_ratio) > target_epsilon:
            high *= 2.0
            if high > 1000.0:
                break

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            eps_mid = cls.compute_rdp_epsilon(mid, num_rounds, delta, subsampling_ratio=subsampling_ratio)

            if abs(eps_mid - target_epsilon) < tol:
                return float(mid)

            if eps_mid > target_epsilon:
                low = mid
            else:
                high = mid

        return float((low + high) / 2.0)
