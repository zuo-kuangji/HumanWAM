from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


class ActionAttentionCapture:
    """Capture action-query attention mass and optional image-token spatial maps.

    Spatial maps are averaged across batch, heads, action queries, and action
    denoising steps.  Only the first ``spatial_max_calls`` inference calls are
    retained so a diagnostic rollout cannot grow the result file without bound.
    """

    def __init__(
        self,
        layers: Iterable[int] | None = None,
        spatial_max_calls: int = 24,
    ):
        self.layers = None if layers is None else {int(layer) for layer in layers}
        self.spatial_max_calls = max(0, int(spatial_max_calls))
        self.reset()

    def reset(self) -> None:
        self.config: dict[str, Any] = {}
        self.inference_call = -1
        self.action_step = 0
        self.records: list[dict[str, Any]] = []
        self.spatial_accumulators: dict[tuple[int, int, str], dict[str, Any]] = {}

    def should_capture(self, layer_idx: int) -> bool:
        return self.layers is None or int(layer_idx) in self.layers

    def configure(
        self,
        *,
        condition_slice,
        condition_grid,
        prefix_len: int,
        action_len: int,
        metadata: dict[str, Any],
    ) -> None:
        self.inference_call += 1
        self.config = {
            **dict(metadata),
            "condition_slice": list(condition_slice),
            "condition_grid": list(condition_grid),
            "prefix_len": int(prefix_len),
            "action_len": int(action_len),
        }

    def start_step(self, step_idx: int) -> None:
        self.action_step = int(step_idx)

    @torch.no_grad()
    def update(
        self,
        attn_probs: torch.Tensor,
        *,
        layer_idx: int,
        block_type: str,
        prefix_len: int,
        action_len: int,
    ) -> None:
        # Average over batch, heads and action queries, leaving a distribution
        # over keys. Its region sums are directly comparable and sum to one.
        key_mass = attn_probs.detach().float().mean(dim=(0, 1, 2)).cpu()
        txt_len = int(self.config.get("txt_len", 0))
        cond_len = int(self.config.get("cond_len", 0))
        target_len = int(self.config.get("target_len", 0))
        current_start = txt_len
        current_end = current_start + cond_len
        future_start = current_end
        future_end = future_start + target_len
        self_start = int(prefix_len)
        self_end = self_start + int(action_len)

        def region(start: int, end: int) -> float:
            if end <= start:
                return 0.0
            return float(key_mass[start:end].sum().item())

        masses = {
            "text": region(0, txt_len),
            "current": region(current_start, current_end),
            "future": region(future_start, future_end),
            "action_self": region(self_start, self_end),
        }
        grid_h, grid_w = (int(x) for x in self.config.get("condition_grid", (0, 0)))
        spatial_len = grid_h * grid_w
        if (
            self.inference_call < self.spatial_max_calls
            and spatial_len > 0
            and cond_len == spatial_len
            and target_len == spatial_len
        ):
            spatial_key = (int(self.inference_call), int(layer_idx), str(block_type))
            current_map = key_mass[current_start:current_end].reshape(grid_h, grid_w)
            future_map = key_mass[future_start:future_end].reshape(grid_h, grid_w)
            accumulator = self.spatial_accumulators.get(spatial_key)
            if accumulator is None:
                accumulator = {
                    "count": 0,
                    "current_sum": torch.zeros_like(current_map),
                    "future_sum": torch.zeros_like(future_map),
                }
                self.spatial_accumulators[spatial_key] = accumulator
            accumulator["count"] += 1
            accumulator["current_sum"] += current_map
            accumulator["future_sum"] += future_map
        self.records.append(
            {
                "inference_call": int(self.inference_call),
                "action_step": int(self.action_step),
                "layer": int(layer_idx),
                "block_type": str(block_type),
                **masses,
                "mass_sum": float(sum(masses.values())),
                "txt_len": txt_len,
                "current_len": cond_len,
                "future_len": target_len,
                "action_len": int(action_len),
            }
        )

    def _spatial_summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        grid_h, grid_w = (int(x) for x in self.config.get("condition_grid", (0, 0)))
        for (inference_call, layer, block_type), item in sorted(self.spatial_accumulators.items()):
            count = int(item["count"])
            current_raw = item["current_sum"] / max(count, 1)
            future_raw = item["future_sum"] / max(count, 1)
            current_mass = float(current_raw.sum().item())
            future_mass = float(future_raw.sum().item())
            current_density = current_raw / max(current_mass, 1e-12)
            future_density = future_raw / max(future_mass, 1e-12)
            rows.append(
                {
                    "inference_call": inference_call,
                    "layer": layer,
                    "block_type": block_type,
                    "action_denoising_steps": count,
                    "grid": [grid_h, grid_w],
                    "current_mass": current_mass,
                    "future_mass": future_mass,
                    "current_density": current_density.tolist(),
                    "future_density": future_density.tolist(),
                }
            )
        return rows

    def summary(self, *, include_spatial: bool = False) -> dict[str, Any]:
        fields = ("text", "current", "future", "action_self", "mass_sum")
        by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in self.records:
            by_layer[int(row["layer"])].append(row)

        per_layer = []
        for layer in sorted(by_layer):
            rows = by_layer[layer]
            item = {
                "layer": layer,
                "block_type": rows[0]["block_type"],
                "calls": len(rows),
            }
            for field in fields:
                item[field] = sum(float(row[field]) for row in rows) / len(rows)
            per_layer.append(item)

        groups = {
            "early_0_7": range(0, 8),
            "middle_8_15": range(8, 16),
            "late_16_24": range(16, 25),
            "double_0_4": range(0, 5),
            "single_5_24": range(5, 25),
        }
        grouped = {}
        indexed = {int(row["layer"]): row for row in per_layer}
        for name, layer_ids in groups.items():
            rows = [indexed[layer] for layer in layer_ids if layer in indexed]
            if not rows:
                continue
            grouped[name] = {
                field: sum(float(row[field]) for row in rows) / len(rows)
                for field in fields
            }
            grouped[name]["layers"] = [int(row["layer"]) for row in rows]

        result = {
            "metadata": dict(self.config),
            "num_inference_calls": int(self.inference_call + 1),
            "num_records": len(self.records),
            "per_layer": per_layer,
            "groups": grouped,
        }
        if include_spatial:
            result["spatial_maps"] = self._spatial_summary()
        return result

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(self.summary(include_spatial=True), indent=2),
            encoding="utf-8",
        )


class ActionKVCausalProbe:
    """Measure final normalized-action changes after layer/region K/V masking."""

    def __init__(self, max_calls: int = 12, executed_steps: int = 12):
        self.max_calls = int(max_calls)
        self.executed_steps = int(executed_steps)
        self.reset()

    def reset(self) -> None:
        self.call_count = 0
        self.records: list[dict[str, Any]] = []
        self.metadata: dict[str, Any] = {}

    def should_probe(self) -> bool:
        return self.call_count < self.max_calls

    def start_call(self, metadata: dict[str, Any]) -> int:
        call_idx = self.call_count
        self.call_count += 1
        self.metadata = dict(metadata)
        return call_idx

    @torch.no_grad()
    def record(
        self,
        *,
        call_idx: int,
        layer: int,
        region: str,
        baseline: torch.Tensor,
        ablated: torch.Tensor,
    ) -> None:
        base = baseline.detach().float().cpu()
        other = ablated.detach().float().cpu()
        diff = other - base
        steps = min(self.executed_steps, int(base.shape[-2]))
        base_exec = base[..., :steps, :]
        diff_exec = diff[..., :steps, :]

        def metrics(x: torch.Tensor, delta: torch.Tensor) -> dict[str, Any]:
            x_flat = x.reshape(-1)
            delta_flat = delta.reshape(-1)
            other_flat = x_flat + delta_flat
            base_rms = float(torch.sqrt(torch.mean(x_flat.square())).item())
            l2_rms = float(torch.sqrt(torch.mean(delta_flat.square())).item())
            cosine = float(
                torch.nn.functional.cosine_similarity(
                    x_flat.unsqueeze(0), other_flat.unsqueeze(0), dim=1, eps=1e-8
                ).item()
            )
            return {
                "l1_mean": float(delta_flat.abs().mean().item()),
                "l2_rms": l2_rms,
                "relative_l2_rms": l2_rms / max(base_rms, 1e-8),
                "cosine": cosine,
                "max_abs": float(delta_flat.abs().max().item()),
            }

        self.records.append(
            {
                "call": int(call_idx),
                "layer": int(layer),
                "region": str(region),
                "full_horizon": metrics(base, diff),
                "executed_prefix": metrics(base_exec, diff_exec),
                "per_dim_l1_executed": diff_exec.abs().mean(dim=tuple(range(diff_exec.ndim - 1))).tolist(),
            }
        )

    def summary(self) -> dict[str, Any]:
        metrics = ("l1_mean", "l2_rms", "relative_l2_rms", "cosine", "max_abs")
        grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
        for row in self.records:
            grouped[(int(row["layer"]), str(row["region"]))].append(row)
        per_layer_region = []
        for (layer, region), rows in sorted(grouped.items()):
            item: dict[str, Any] = {"layer": layer, "region": region, "calls": len(rows)}
            for scope in ("full_horizon", "executed_prefix"):
                item[scope] = {
                    metric: sum(float(row[scope][metric]) for row in rows) / len(rows)
                    for metric in metrics
                }
            dims = len(rows[0]["per_dim_l1_executed"])
            item["per_dim_l1_executed"] = [
                sum(float(row["per_dim_l1_executed"][dim]) for row in rows) / len(rows)
                for dim in range(dims)
            ]
            per_layer_region.append(item)
        return {
            "metadata": dict(self.metadata),
            "num_calls": self.call_count,
            "num_records": len(self.records),
            "per_layer_region": per_layer_region,
        }

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
