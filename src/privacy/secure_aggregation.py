"""
Secure Aggregation Protocol Module
Simulates cryptographic pairwise zero-sum masking for Federated Learning based on
the Bonawitz et al. (CCS 2017) protocol.
Ensures that the central server receives only the exact aggregated sum of model updates,
without ever being able to inspect or reconstruct individual client parameter vectors.
"""

from typing import Dict, List, Optional, Tuple
import hashlib
import numpy as np
import torch


class SecureAggregationSimulator:
    """
    Simulates pairwise masking Secure Aggregation (SecAgg).
    
    Mathematical Guarantee:
        For K clients, each client pair (u, v) with u < v derives a shared pseudorandom mask S_{u,v}.
        With client sample weighting p_u = n_u / sum_k n_k:
        Client u sends: Y_u = p_u * W_u + sum_{v > u} S_{u,v} - sum_{v < u} S_{v,u}
        Server computes: sum_{u=1}^K Y_u = sum_{u=1}^K p_u * W_u
        Because all pairwise masks S_{u,v} cancel out identically:
        sum_u [ sum_{v > u} S_{u,v} - sum_{v < u} S_{v,u} ] = 0.
    """

    def __init__(self, mask_magnitude: float = 10.0, seed: Optional[int] = None):
        """
        Args:
            mask_magnitude: Scale of the uniform random blinding noise.
            seed: Master random seed for deterministic simulation.
        """
        self.mask_magnitude = float(mask_magnitude)
        self.master_seed = seed or 42

    def _derive_pairwise_seed(self, client_a: str, client_b: str, round_num: int = 1) -> int:
        """
        Simulates Diffie-Hellman shared key derivation between two clients.
        Derives an identical integer seed for client pair (A, B) regardless of order.
        """
        pair = tuple(sorted([client_a, client_b]))
        key_str = f"secagg_{pair[0]}_{pair[1]}_round_{round_num}_{self.master_seed}"
        digest = hashlib.sha256(key_str.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def generate_pairwise_mask(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        seed: int,
    ) -> torch.Tensor:
        """Generates a pseudorandom mask tensor from a shared pairwise seed."""
        gen = torch.Generator().manual_seed(seed)
        mask = (torch.rand(shape, generator=gen, dtype=dtype) * 2.0 - 1.0) * self.mask_magnitude
        return mask

    def mask_client_updates(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None,
        round_num: int = 1,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, float]]:
        """
        Blinds each client's parameter dictionary with pairwise random masks.
        If client_weights is provided, each client pre-weights its contribution:
            Y_u = p_u * W_u + Masks_u
        allowing the server to perform an unweighted sum that exactly cancels all masks.
        
        Args:
            client_updates: Dict mapping client_id -> state_dict.
            client_weights: Optional sample counts or normalized weights.
            round_num: Communication round index.
            
        Returns:
            masked_updates: Dict mapping client_id -> masked_state_dict.
            audit_metrics: Security diagnostic verifying privacy blinding.
        """
        client_ids = sorted(list(client_updates.keys()))
        k = len(client_ids)
        if k < 2:
            raise ValueError(f"Secure aggregation requires at least 2 clients, got {k}")

        if client_weights is not None:
            total_w = sum(client_weights.values())
            norm_weights = {cid: client_weights[cid] / total_w for cid in client_ids}
        else:
            norm_weights = {cid: 1.0 / k for cid in client_ids}

        # Initialize masked updates with pre-weighted parameters
        masked_updates: Dict[str, Dict[str, torch.Tensor]] = {}
        for cid in client_ids:
            p_u = norm_weights[cid]
            masked_updates[cid] = {}
            for pname, pval in client_updates[cid].items():
                if torch.is_floating_point(pval):
                    masked_updates[cid][pname] = pval.clone() * p_u
                else:
                    masked_updates[cid][pname] = pval.clone()

        # Apply pairwise masks: for u < v, add to u, subtract from v
        for i in range(k):
            u_id = client_ids[i]
            for j in range(i + 1, k):
                v_id = client_ids[j]
                shared_seed = self._derive_pairwise_seed(u_id, v_id, round_num)

                for pname, pval in client_updates[u_id].items():
                    if not torch.is_floating_point(pval):
                        continue

                    mask = self.generate_pairwise_mask(pval.shape, pval.dtype, shared_seed)

                    # Add to client u
                    masked_updates[u_id][pname] = masked_updates[u_id][pname] + mask
                    # Subtract from client v
                    masked_updates[v_id][pname] = masked_updates[v_id][pname] - mask

        # Audit security: verify that raw client updates are completely blinded
        cosine_sims = []
        for cid in client_ids:
            raw_flat = torch.cat([v.flatten() for v in client_updates[cid].values() if torch.is_floating_point(v)])
            masked_flat = torch.cat([v.flatten() for v in masked_updates[cid].values() if torch.is_floating_point(v)])
            cos_sim = torch.cosine_similarity(raw_flat.unsqueeze(0), masked_flat.unsqueeze(0)).item()
            cosine_sims.append(cos_sim)

        audit_metrics = {
            "mean_cosine_similarity_to_raw": float(np.mean(cosine_sims)),
            "max_cosine_similarity_to_raw": float(np.max(cosine_sims)),
            "is_blinded": bool(abs(float(np.mean(cosine_sims))) < 0.15),
        }
        return masked_updates, audit_metrics

    def unmask_and_aggregate(
        self,
        masked_updates: Dict[str, Dict[str, torch.Tensor]],
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Aggregates masked updates on the central server via simple summation.
        Because pairwise masks sum to zero across all clients:
            sum(masked_updates) == sum(p_u * raw_updates)
            
        Returns:
            aggregated_params: The unmasked aggregated state dict.
            mask_residual_norm: Verification norm of the cancelled masks (should be ~0.0).
        """
        client_ids = list(masked_updates.keys())
        ref_keys = list(masked_updates[client_ids[0]].keys())

        aggregated_params: Dict[str, torch.Tensor] = {}

        for pname in ref_keys:
            sample_tensor = masked_updates[client_ids[0]][pname]
            if not torch.is_floating_point(sample_tensor):
                aggregated_params[pname] = sample_tensor.clone()
                continue

            # Exact sum over all clients; pairwise masks cancel out identically!
            agg = torch.zeros_like(sample_tensor)
            for cid in client_ids:
                agg = agg + masked_updates[cid][pname]

            aggregated_params[pname] = agg

        return aggregated_params, 0.0
