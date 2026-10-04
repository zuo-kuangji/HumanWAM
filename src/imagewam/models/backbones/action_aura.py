"""Action-side AURA components shared by training and rollout inference.

This module deliberately does not alter the visual condition consumed by
ImageWAM.  ScoNet reads a detached, pooled representation of the current FLUX.2
image tokens and controls only the action flow source and action schedules.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def action_source_is_stochastic(
    *,
    action_init_present: bool,
    action_init_noise_strength: float,
    action_aura_enabled: bool,
) -> bool:
    """Return whether rollout inference samples a random action-flow source.

    The default Gaussian baseline has no explicit ``action_init`` tensor, but
    ImageWAM still samples its source with ``torch.randn``.  Explicit history
    is deterministic only when neither action-init noise nor AURA is enabled.
    """
    return (
        not bool(action_init_present)
        or float(action_init_noise_strength) > 0.0
        or bool(action_aura_enabled)
    )


def resolve_action_source_seed(
    base_seed: int | None,
    *,
    source_is_stochastic: bool,
    reseed_each_call: bool,
) -> int | None:
    """Choose between a fixed per-call seed and the advancing global RNG.

    AURA must draw fresh source noise after every replan.  Returning ``None``
    lets the already seeded process RNG advance across calls while preserving
    reproducibility for a complete evaluation run.
    """
    if source_is_stochastic and not reseed_each_call:
        return None
    return None if base_seed is None else int(base_seed)


class ActionAuraScoNet(nn.Module):
    """Lightweight per-action-dimension uncertainty router."""

    def __init__(
        self,
        *,
        image_token_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        pool_tokens: int = 8,
    ) -> None:
        super().__init__()
        if image_token_dim <= 0 or action_dim <= 0 or hidden_dim <= 0 or pool_tokens <= 0:
            raise ValueError("ScoNet dimensions and pool_tokens must all be positive.")
        self.image_token_dim = int(image_token_dim)
        self.action_dim = int(action_dim)
        self.pool_tokens = int(pool_tokens)
        input_dim = self.image_token_dim * self.pool_tokens
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.action_dim),
        )

    def forward(self, current_image_tokens: torch.Tensor) -> torch.Tensor:
        if current_image_tokens.ndim != 3:
            raise ValueError(
                "ScoNet expects current image tokens [B,N,C], got "
                f"{tuple(current_image_tokens.shape)}."
            )
        if int(current_image_tokens.shape[-1]) != self.image_token_dim:
            raise ValueError(
                f"ScoNet image token dim must be {self.image_token_dim}, got "
                f"{int(current_image_tokens.shape[-1])}."
            )
        # Keep coarse spatial/camera layout instead of collapsing the current
        # image to one global mean. The visual tokens are detached by the caller.
        pooled = F.adaptive_avg_pool1d(
            current_image_tokens.transpose(1, 2).float(), self.pool_tokens
        ).flatten(1)
        logits = self.net(pooled.to(dtype=self.net[1].weight.dtype))
        return torch.sigmoid(logits)


def action_dim_mask(
    action_dim: int,
    dims: Sequence[int] | None,
    *,
    device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(int(action_dim), dtype=torch.bool, device=device)
    for dim in dims or ():
        dim = int(dim)
        if dim < 0:
            dim += int(action_dim)
        if dim < 0 or dim >= int(action_dim):
            raise ValueError(f"Action dimension {dim} is out of range for D={action_dim}.")
        mask[dim] = True
    return mask


def finalize_router_weight(
    weight: torch.Tensor,
    *,
    noise_dims: Sequence[int] | None,
) -> torch.Tensor:
    """Pin configured non-continuous dimensions to a pure-Gaussian source."""
    if not noise_dims:
        return weight
    mask = action_dim_mask(weight.shape[-1], noise_dims, device=weight.device)
    return torch.where(mask.view(1, -1), torch.ones_like(weight), weight)


def router_scalar(
    weight: torch.Tensor,
    *,
    noise_dims: Sequence[int] | None,
) -> torch.Tensor:
    """Conservative per-sample uncertainty, excluding pinned noise dimensions."""
    if weight.ndim != 2:
        raise ValueError(f"Router weight must be [B,D], got {tuple(weight.shape)}.")
    if not noise_dims:
        return weight.amax(dim=-1)
    mask = action_dim_mask(weight.shape[-1], noise_dims, device=weight.device)
    if bool(mask.all().item()):
        raise ValueError("AURA noise_dims cannot cover every action dimension.")
    return weight.masked_fill(mask.view(1, -1), float("-inf")).amax(dim=-1)


def mix_action_source(
    history_action: torch.Tensor,
    weight: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    rand_device: str | torch.device | None = None,
) -> torch.Tensor:
    """Apply x0=(1-w)h+w*epsilon with w shared over the action horizon."""
    if history_action.ndim != 3 or weight.ndim != 2:
        raise ValueError(
            "AURA source expects history [B,T,D] and weight [B,D], got "
            f"{tuple(history_action.shape)} and {tuple(weight.shape)}."
        )
    if history_action.shape[0] != weight.shape[0] or history_action.shape[-1] != weight.shape[-1]:
        raise ValueError("AURA history and router weight batch/action dimensions must match.")
    if generator is None and rand_device is None:
        noise = torch.randn_like(history_action)
    else:
        noise = torch.randn(
            tuple(history_action.shape),
            generator=generator,
            device=rand_device or history_action.device,
            dtype=torch.float32,
        ).to(device=history_action.device, dtype=history_action.dtype)
    w = weight.to(device=history_action.device, dtype=history_action.dtype).unsqueeze(1)
    return (1.0 - w) * history_action + w * noise


def adaptive_inference_steps(
    weight: torch.Tensor,
    *,
    max_steps: int,
    noise_dims: Sequence[int] | None,
) -> torch.Tensor:
    if int(max_steps) <= 0:
        raise ValueError(f"AURA max inference steps must be positive, got {max_steps}.")
    score = router_scalar(weight, noise_dims=noise_dims).clamp(0.0, 1.0)
    return torch.ceil(score * int(max_steps)).clamp(min=1, max=int(max_steps)).long()


def adaptive_execution_steps(
    weight: torch.Tensor,
    *,
    min_steps: int,
    max_steps: int,
    noise_dims: Sequence[int] | None,
) -> torch.Tensor:
    min_steps, max_steps = int(min_steps), int(max_steps)
    if min_steps <= 0 or max_steps < min_steps:
        raise ValueError(
            f"AURA execution bounds require 0 < min <= max, got {min_steps}, {max_steps}."
        )
    score = router_scalar(weight, noise_dims=noise_dims).clamp(0.0, 1.0)
    steps = min_steps + (max_steps - min_steps) * (1.0 - score)
    return steps.round().clamp(min=min_steps, max=max_steps).long()


def diversity_loss_global_knn(
    start: torch.Tensor,
    target_spread: torch.Tensor,
    nn_histories: torch.Tensor,
    weight: torch.Tensor,
    *,
    noise_dims: Sequence[int] | None,
    nn_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """AURA per-dimension source-spread deficit using precomputed visual KNNs."""
    if start.ndim != 3 or target_spread.ndim != 2 or nn_histories.ndim != 4:
        raise ValueError(
            "AURA diversity expects start [B,T,D], target_spread [B,D], and "
            f"nn_histories [B,K,T,D], got {tuple(start.shape)}, "
            f"{tuple(target_spread.shape)}, {tuple(nn_histories.shape)}."
        )
    batch, horizon, action_dim = start.shape
    expected = (batch, action_dim)
    if tuple(target_spread.shape) != expected or tuple(weight.shape) != expected:
        raise ValueError(
            f"AURA diversity target_spread/weight must both be {expected}, got "
            f"{tuple(target_spread.shape)} and {tuple(weight.shape)}."
        )
    if nn_histories.shape[0] != batch or nn_histories.shape[2:] != (horizon, action_dim):
        raise ValueError("AURA KNN histories must match start batch, horizon, and action dim.")

    nn_hist = nn_histories.detach().to(device=start.device, dtype=start.dtype)
    w = weight.view(batch, 1, 1, action_dim).to(dtype=start.dtype)
    nn_start = (1.0 - w) * nn_hist + w * torch.randn_like(nn_hist)
    per_pair = (start.unsqueeze(1) - nn_start).abs().mean(dim=2)
    if nn_weights is None:
        source_spread = per_pair.mean(dim=1)
    else:
        weights = nn_weights.detach().to(device=start.device, dtype=per_pair.dtype)
        if tuple(weights.shape) != tuple(per_pair.shape[:2]):
            raise ValueError(
                f"AURA nn_weights must be {tuple(per_pair.shape[:2])}, got {tuple(weights.shape)}."
            )
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        source_spread = (weights.unsqueeze(-1) * per_pair).sum(dim=1)

    gap = F.relu(target_spread.detach().to(device=start.device, dtype=source_spread.dtype) - source_spread)
    if not noise_dims:
        return gap.mean()
    active = ~action_dim_mask(action_dim, noise_dims, device=start.device)
    return gap[:, active].mean()
