"""
Differential Privacy Engine Module
Provides Local Differential Privacy (LDP / Model Update Perturbation) for edge nodes.
Implements global L2-norm update clipping and calibrated Gaussian noise injection.
"""

from typing import Dict, Optional, Tuple
import math
import torch


class DifferentialPrivacyEngine:
    """
    Applies Differential Privacy mechanisms to model parameter updates.
    
    Guarantees that individual edge client parameter updates satisfy bounded sensitivity
    via L2-norm clipping, followed by Gaussian mechanism perturbation:
        Delta_w = w_local - w_global
        Delta_w_clipped = Delta_w * min(1, C / ||Delta_w||_2)
        Delta_w_priv = Delta_w_clipped + N(0, (sigma * C)^2 * I)
        w_priv = w_global + Delta_w_priv
    """

    def __init__(
        self,
        clip_norm: float = 1.0,
        noise_multiplier: float = 0.05,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Args:
            clip_norm (C): Maximum allowed L2 norm of the parameter update.
            noise_multiplier (sigma): Noise standard deviation ratio relative to clipping threshold (sigma * C).
            device: Computing device ('cpu' or 'cuda').
            seed: Optional random seed for reproducible noise generation.
        """
        if clip_norm <= 0:
            raise ValueError(f"clip_norm must be positive, got {clip_norm}")
        if noise_multiplier < 0:
            raise ValueError(f"noise_multiplier must be non-negative, got {noise_multiplier}")

        self.clip_norm = float(clip_norm)
        self.noise_multiplier = float(noise_multiplier)
        self.device = device
        self.rng = torch.Generator(device=device)
        if seed is not None:
            self.rng.manual_seed(seed)

    def compute_update_norm(
        self,
        local_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
    ) -> float:
        """
        Computes the global L2 norm of the parameter update vector across all floating-point layers:
        ||Delta_w||_2 = sqrt( sum_p ||w_local,p - w_global,p||_2^2 )
        """
        total_sq_norm = 0.0
        for name, local_val in local_params.items():
            if name in global_params and torch.is_floating_point(local_val):
                diff = local_val.to(self.device) - global_params[name].to(self.device)
                total_sq_norm += diff.pow(2).sum().item()
        return math.sqrt(total_sq_norm)

    def clip_and_perturb_update(
        self,
        local_params: Dict[str, torch.Tensor],
        global_params: Dict[str, torch.Tensor],
        custom_clip_norm: Optional[float] = None,
        custom_noise_multiplier: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Clips the parameter update vector to clip_norm and adds calibrated Gaussian noise.
        
        Returns:
            privatized_params: Sanitized model weights ready for secure broadcast/aggregation.
            stats: Diagnostic dictionary containing pre_clip_norm, post_clip_norm, and noise_std.
        """
        c = custom_clip_norm if custom_clip_norm is not None else self.clip_norm
        sigma = custom_noise_multiplier if custom_noise_multiplier is not None else self.noise_multiplier

        # 1. Compute pre-clipping global L2 norm
        raw_norm = self.compute_update_norm(local_params, global_params)
        clip_factor = min(1.0, c / (raw_norm + 1e-12))

        # 2. Compute noise standard deviation: sigma * C
        noise_std = sigma * c

        privatized_params = {}
        for name, local_val in local_params.items():
            if name not in global_params:
                privatized_params[name] = local_val.clone()
                continue

            global_val = global_params[name].to(self.device)
            local_val_dev = local_val.to(self.device)

            if not torch.is_floating_point(local_val):
                # Keep non-floating point parameters (e.g., integer buffers) unchanged
                privatized_params[name] = local_val.cpu().clone()
                continue

            # Compute difference vector
            delta = local_val_dev - global_val

            # Apply L2 clipping
            delta_clipped = delta * clip_factor

            # Inject calibrated Gaussian noise
            if noise_std > 0.0:
                noise = torch.normal(
                    mean=0.0,
                    std=noise_std,
                    size=delta.shape,
                    generator=self.rng,
                    device=self.device,
                    dtype=delta.dtype,
                )
                delta_priv = delta_clipped + noise
            else:
                delta_priv = delta_clipped

            # Reconstruct privatized parameter tensor
            w_priv = global_val + delta_priv
            privatized_params[name] = w_priv.cpu().clone()

        post_clip_norm = min(raw_norm, c)
        stats = {
            "pre_clip_norm": float(raw_norm),
            "clip_factor": float(clip_factor),
            "post_clip_norm": float(post_clip_norm),
            "noise_std": float(noise_std),
        }
        return privatized_params, stats
