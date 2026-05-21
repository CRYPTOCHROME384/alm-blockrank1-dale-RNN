from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


DELAY_SHAPE_LOSS_NONE = "none"
DELAY_SHAPE_LOSS_RAW = "raw_centered_mse"
DELAY_SHAPE_LOSS_REL_GLOBAL = "relative_ac_global"
DELAY_SHAPE_LOSS_REL_GROUP = "relative_ac_group"
SUPPORTED_DELAY_SHAPE_LOSS_TYPES = {
    DELAY_SHAPE_LOSS_NONE,
    DELAY_SHAPE_LOSS_RAW,
    DELAY_SHAPE_LOSS_REL_GLOBAL,
    DELAY_SHAPE_LOSS_REL_GROUP,
}
DEFAULT_DELAY_SHAPE_EPS = 1e-8
DEFAULT_DELAY_SHAPE_MIN_SCALE = 1e-6


def _as_bool_mask(delay_mask: Any, *, device: torch.device) -> torch.Tensor:
    mask_np = np.asarray(delay_mask, dtype=np.bool_)
    if mask_np.ndim != 1:
        raise ValueError(f"delay_mask must be 1D, got shape={tuple(mask_np.shape)}")
    return torch.as_tensor(mask_np, dtype=torch.bool, device=device)


def _normalize_loss_type(loss_type: Optional[str]) -> str:
    out = str(loss_type or DELAY_SHAPE_LOSS_NONE).strip().lower()
    if out not in SUPPORTED_DELAY_SHAPE_LOSS_TYPES:
        raise ValueError(f"Unsupported delay_shape_loss_type={loss_type!r}; expected one of {sorted(SUPPORTED_DELAY_SHAPE_LOSS_TYPES)}")
    return out


def _normalize_group_mode(group_mode: Optional[str]) -> str:
    out = str(group_mode or "all").strip().lower()
    if out not in {"all", "celltype"}:
        raise ValueError(f"Unsupported delay_shape_group_mode={group_mode!r}; expected 'all' or 'celltype'")
    return out


def extract_delay_segment(x: torch.Tensor, delay_mask: Any) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"x must have shape [C,T,K], got {tuple(x.shape)}")
    mask_t = _as_bool_mask(delay_mask, device=x.device)
    if int(mask_t.sum().item()) <= 0:
        raise ValueError("delay_mask selects zero frames.")
    return x[:, mask_t, :]


def center_over_delay_time(x: torch.Tensor, delay_mask: Any) -> Dict[str, torch.Tensor]:
    seg = extract_delay_segment(x, delay_mask)
    mean_t = seg.mean(dim=1, keepdim=True)
    centered = seg - mean_t
    max_abs_mean = centered.mean(dim=1).abs().max()
    return {
        "segment": seg,
        "mean_t": mean_t,
        "centered": centered,
        "max_abs_mean": max_abs_mean,
    }


def build_delay_group_info(
    *,
    K: int,
    group_mode: str = "all",
    group_index: Optional[Any] = None,
    group_names: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    group_mode_norm = _normalize_group_mode(group_mode)
    dev = device if device is not None else torch.device("cpu")

    if group_mode_norm == "all":
        return {
            "group_mode": "all",
            "group_labels_by_neuron": ["all"] * int(K),
            "group_names": ["all"],
            "group_masks": {"all": torch.ones(int(K), dtype=torch.bool, device=dev)},
            "group_sizes": {"all": int(K)},
        }

    if group_index is None:
        raise ValueError("group_index is required when delay_shape_group_mode='celltype'")
    group_index_np = np.asarray(group_index)
    if group_index_np.ndim != 1 or int(group_index_np.shape[0]) != int(K):
        raise ValueError(f"group_index must be 1D with length K={K}, got shape={tuple(group_index_np.shape)}")

    if group_names is not None:
        group_labels = [str(group_names[int(i)]) for i in group_index_np.astype(np.int64).tolist()]
    else:
        group_labels = [str(x) for x in group_index_np.tolist()]

    unique_labels = sorted({str(x) for x in group_labels})
    group_masks = {
        str(name): torch.as_tensor(np.asarray([lbl == str(name) for lbl in group_labels], dtype=np.bool_), dtype=torch.bool, device=dev)
        for name in unique_labels
    }
    group_sizes = {name: int(mask.sum().item()) for name, mask in group_masks.items()}
    return {
        "group_mode": group_mode_norm,
        "group_labels_by_neuron": list(group_labels),
        "group_names": list(unique_labels),
        "group_masks": group_masks,
        "group_sizes": group_sizes,
    }


def delay_centered_shape_loss(target: torch.Tensor, pred: torch.Tensor, delay_mask: Any) -> torch.Tensor:
    if tuple(target.shape) != tuple(pred.shape):
        raise ValueError(f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}")
    tgt = center_over_delay_time(target, delay_mask)["centered"]
    prd = center_over_delay_time(pred, delay_mask)["centered"]
    return (prd - tgt).pow(2).mean()


def delay_derivative_loss(target: torch.Tensor, pred: torch.Tensor, delay_mask: Any) -> torch.Tensor:
    if tuple(target.shape) != tuple(pred.shape):
        raise ValueError(f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}")
    tgt = extract_delay_segment(target, delay_mask)
    prd = extract_delay_segment(pred, delay_mask)
    if int(tgt.shape[1]) < 2:
        return tgt.new_zeros(())
    dt_tgt = tgt[:, 1:, :] - tgt[:, :-1, :]
    dt_prd = prd[:, 1:, :] - prd[:, :-1, :]
    return (dt_prd - dt_tgt).pow(2).mean()


def delay_epoch_normalized_mse(
    target: torch.Tensor,
    pred: torch.Tensor,
    delay_mask: Any,
    *,
    eps: float = DEFAULT_DELAY_SHAPE_EPS,
) -> torch.Tensor:
    seg_tgt = extract_delay_segment(target, delay_mask)
    seg_prd = extract_delay_segment(pred, delay_mask)
    mse = (seg_prd - seg_tgt).pow(2).mean()
    scale = seg_tgt.pow(2).mean().detach().clamp_min(float(eps))
    return mse / scale


def make_constant_delay_baseline(target: torch.Tensor, delay_mask: Any) -> torch.Tensor:
    seg_info = center_over_delay_time(target, delay_mask)
    pred = target.clone()
    mask_t = _as_bool_mask(delay_mask, device=target.device)
    pred[:, mask_t, :] = seg_info["mean_t"].expand_as(seg_info["segment"])
    return pred


def compute_relative_ac_loss(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    delay_mask: Any,
    scale_mode: str = "global_ac_energy",
    group_mode: str = "all",
    group_index: Optional[Any] = None,
    group_names: Optional[Sequence[str]] = None,
    eps: float = DEFAULT_DELAY_SHAPE_EPS,
    min_scale: float = DEFAULT_DELAY_SHAPE_MIN_SCALE,
    min_group_points: int = 4,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if tuple(target.shape) != tuple(pred.shape):
        raise ValueError(f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}")
    if str(scale_mode) not in {"global_ac_energy", "group_ac_energy"}:
        raise ValueError(f"Unsupported scale_mode={scale_mode!r}")

    tgt_info = center_over_delay_time(target, delay_mask)
    prd_info = center_over_delay_time(pred, delay_mask)
    target_ac = tgt_info["centered"]
    pred_ac = prd_info["centered"]
    diff_sq = (pred_ac - target_ac).pow(2)
    target_ac_sq = target_ac.pow(2)

    details: Dict[str, Any] = {
        "scale_mode": str(scale_mode),
        "group_mode": _normalize_group_mode(group_mode),
        "eps": float(eps),
        "min_scale": float(min_scale),
        "warnings": [],
        "groups": [],
    }

    if str(scale_mode) == "global_ac_energy":
        scale_raw = target_ac_sq.mean()
        scale_used = scale_raw.detach().clamp_min(max(float(min_scale), float(eps)))
        loss = diff_sq.mean() / scale_used
        if float(scale_raw.detach().cpu().item()) < float(min_scale):
            details["warnings"].append(
                f"global_ac_energy scale_raw={float(scale_raw.detach().cpu().item()):.6e} below min_scale={float(min_scale):.6e}; clamped."
            )
        details["scale_raw"] = float(scale_raw.detach().cpu().item())
        details["scale_used"] = float(scale_used.detach().cpu().item())
        details["numerator_mean"] = float(diff_sq.mean().detach().cpu().item())
        details["target_ac_energy"] = float(target_ac_sq.mean().detach().cpu().item())
        details["group_names"] = ["all"]
        return loss, details

    group_info = build_delay_group_info(
        K=int(target.shape[-1]),
        group_mode=str(group_mode),
        group_index=group_index,
        group_names=group_names,
        device=target.device,
    )
    losses: List[torch.Tensor] = []
    active_names: List[str] = []
    for name in group_info["group_names"]:
        mask = group_info["group_masks"][str(name)]
        n_points = int(mask.sum().item()) * int(target_ac.shape[0]) * int(target_ac.shape[1])
        if int(mask.sum().item()) <= 0:
            details["warnings"].append(f"group={name} has zero neurons; skipped.")
            continue
        diff_sq_g = diff_sq[:, :, mask]
        target_ac_sq_g = target_ac_sq[:, :, mask]
        numerator = diff_sq_g.mean()
        scale_raw = target_ac_sq_g.mean()
        scale_used = scale_raw.detach().clamp_min(max(float(min_scale), float(eps)))
        loss_g = numerator / scale_used
        if n_points < int(min_group_points):
            details["warnings"].append(f"group={name} has only n_points={n_points}; results may be noisy.")
        if float(scale_raw.detach().cpu().item()) < float(min_scale):
            details["warnings"].append(
                f"group={name} scale_raw={float(scale_raw.detach().cpu().item()):.6e} below min_scale={float(min_scale):.6e}; clamped."
            )
        details["groups"].append(
            {
                "group_name": str(name),
                "n_neurons": int(mask.sum().item()),
                "n_points": int(n_points),
                "scale_raw": float(scale_raw.detach().cpu().item()),
                "scale_used": float(scale_used.detach().cpu().item()),
                "numerator_mean": float(numerator.detach().cpu().item()),
                "target_ac_energy": float(target_ac_sq_g.mean().detach().cpu().item()),
                "loss_group": float(loss_g.detach().cpu().item()),
            }
        )
        losses.append(loss_g)
        active_names.append(str(name))

    if len(losses) == 0:
        fallback = diff_sq.mean() / target_ac_sq.mean().detach().clamp_min(max(float(min_scale), float(eps)))
        details["warnings"].append("No valid groups found; fell back to global relative AC loss.")
        details["group_names"] = []
        details["scale_raw"] = float(target_ac_sq.mean().detach().cpu().item())
        details["scale_used"] = float(target_ac_sq.mean().detach().clamp_min(max(float(min_scale), float(eps))).cpu().item())
        return fallback, details

    loss = torch.stack(losses).mean()
    details["group_names"] = list(active_names)
    details["n_groups_used"] = int(len(losses))
    return loss, details


def compute_delay_shape_loss_bundle(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    delay_mask: Any,
    loss_type: str = DELAY_SHAPE_LOSS_NONE,
    eps: float = DEFAULT_DELAY_SHAPE_EPS,
    min_scale: float = DEFAULT_DELAY_SHAPE_MIN_SCALE,
    group_mode: str = "all",
    group_index: Optional[Any] = None,
    group_names: Optional[Sequence[str]] = None,
    min_group_points: int = 4,
) -> Dict[str, Any]:
    loss_type_norm = _normalize_loss_type(loss_type)
    out: Dict[str, Any] = {
        "loss_type": str(loss_type_norm),
        "loss": target.new_zeros(()),
        "details": {
            "loss_type": str(loss_type_norm),
            "group_mode": _normalize_group_mode(group_mode),
            "eps": float(eps),
            "min_scale": float(min_scale),
            "warnings": [],
        },
    }
    if loss_type_norm == DELAY_SHAPE_LOSS_NONE:
        return out
    if loss_type_norm == DELAY_SHAPE_LOSS_RAW:
        loss = delay_centered_shape_loss(target, pred, delay_mask)
        out["loss"] = loss
        out["details"]["numerator_mean"] = float(loss.detach().cpu().item())
        return out
    if loss_type_norm == DELAY_SHAPE_LOSS_REL_GLOBAL:
        loss, details = compute_relative_ac_loss(
            target=target,
            pred=pred,
            delay_mask=delay_mask,
            scale_mode="global_ac_energy",
            group_mode="all",
            eps=float(eps),
            min_scale=float(min_scale),
            min_group_points=int(min_group_points),
        )
        out["loss"] = loss
        out["details"] = details
        return out
    if loss_type_norm == DELAY_SHAPE_LOSS_REL_GROUP:
        loss, details = compute_relative_ac_loss(
            target=target,
            pred=pred,
            delay_mask=delay_mask,
            scale_mode="group_ac_energy",
            group_mode=str(group_mode),
            group_index=group_index,
            group_names=group_names,
            eps=float(eps),
            min_scale=float(min_scale),
            min_group_points=int(min_group_points),
        )
        out["loss"] = loss
        out["details"] = details
        return out
    raise AssertionError(f"Unhandled delay_shape_loss_type={loss_type_norm!r}")


def compute_delay_statistics(x: torch.Tensor, delay_mask: Any) -> Dict[str, torch.Tensor]:
    seg_info = center_over_delay_time(x, delay_mask)
    seg = seg_info["segment"]
    centered = seg_info["centered"]
    std = seg.std(dim=1, unbiased=False)
    centered_energy = centered.pow(2).mean(dim=1)
    mean_abs = seg.abs().mean(dim=1)
    if int(seg.shape[1]) >= 2:
        deriv = seg[:, 1:, :] - seg[:, :-1, :]
        derivative_rms = deriv.pow(2).mean(dim=1).sqrt()
    else:
        derivative_rms = torch.zeros_like(std)
    return {
        "std": std,
        "derivative_rms": derivative_rms,
        "centered_energy": centered_energy,
        "mean_abs": mean_abs,
        "max_abs_centered_mean_over_time": seg_info["max_abs_mean"],
    }


def compute_delay_component_errors(target: torch.Tensor, pred: torch.Tensor, delay_mask: Any) -> Dict[str, torch.Tensor]:
    if tuple(target.shape) != tuple(pred.shape):
        raise ValueError(f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}")
    tgt_info = center_over_delay_time(target, delay_mask)
    prd_info = center_over_delay_time(pred, delay_mask)
    seg_tgt = tgt_info["segment"]
    seg_prd = prd_info["segment"]
    dc_error = (prd_info["mean_t"] - tgt_info["mean_t"]).pow(2).squeeze(dim=1)
    ac_error = (prd_info["centered"] - tgt_info["centered"]).pow(2).mean(dim=1)
    total_delay_mse = (seg_prd - seg_tgt).pow(2).mean(dim=1)
    return {
        "delay_mse": total_delay_mse,
        "dc_error": dc_error,
        "ac_error": ac_error,
        "shape_loss": ac_error,
    }


def summarize_condition_neuron_metrics(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    delay_mask: Any,
    group_mode: str = "all",
    group_index: Optional[Any] = None,
    group_names: Optional[Sequence[str]] = None,
    eps: float = DEFAULT_DELAY_SHAPE_EPS,
    min_scale: float = DEFAULT_DELAY_SHAPE_MIN_SCALE,
) -> Dict[str, Any]:
    tgt_stats = compute_delay_statistics(target, delay_mask)
    prd_stats = compute_delay_statistics(pred, delay_mask)
    errs = compute_delay_component_errors(target, pred, delay_mask)
    rel_global = compute_delay_shape_loss_bundle(
        target=target,
        pred=pred,
        delay_mask=delay_mask,
        loss_type=DELAY_SHAPE_LOSS_REL_GLOBAL,
        eps=float(eps),
        min_scale=float(min_scale),
    )
    rel_group = compute_delay_shape_loss_bundle(
        target=target,
        pred=pred,
        delay_mask=delay_mask,
        loss_type=DELAY_SHAPE_LOSS_REL_GROUP,
        group_mode=str(group_mode),
        group_index=group_index,
        group_names=group_names,
        eps=float(eps),
        min_scale=float(min_scale),
    )

    tgt_std = tgt_stats["std"]
    prd_std = prd_stats["std"]
    std_ratio = torch.full_like(tgt_std, float("nan"))
    valid_std = tgt_std.abs() > 1e-12
    std_ratio[valid_std] = prd_std[valid_std] / tgt_std[valid_std]

    tgt_mean_abs = tgt_stats["mean_abs"]
    prd_mean_abs = prd_stats["mean_abs"]
    mean_abs_ratio = torch.full_like(tgt_mean_abs, float("nan"))
    valid_abs = tgt_mean_abs.abs() > 1e-12
    mean_abs_ratio[valid_abs] = prd_mean_abs[valid_abs] / tgt_mean_abs[valid_abs]

    return {
        "target_delay_std": tgt_stats["std"],
        "target_delay_derivative_rms": tgt_stats["derivative_rms"],
        "target_delay_centered_energy": tgt_stats["centered_energy"],
        "target_delay_mean_abs": tgt_stats["mean_abs"],
        "pred_delay_std": prd_stats["std"],
        "pred_delay_derivative_rms": prd_stats["derivative_rms"],
        "pred_delay_centered_energy": prd_stats["centered_energy"],
        "pred_delay_mean_abs": prd_stats["mean_abs"],
        "delay_mse": errs["delay_mse"],
        "dc_error": errs["dc_error"],
        "ac_error": errs["ac_error"],
        "delay_shape_loss": errs["shape_loss"],
        "delay_ac_rel_global_loss": rel_global["loss"],
        "delay_ac_rel_group_loss": rel_group["loss"],
        "delay_ac_rel_global_details": rel_global["details"],
        "delay_ac_rel_group_details": rel_group["details"],
        "delay_std_ratio": std_ratio,
        "delay_mean_abs_ratio": mean_abs_ratio,
        "centered_mean_maxabs_target": tgt_stats["max_abs_centered_mean_over_time"],
        "centered_mean_maxabs_pred": prd_stats["max_abs_centered_mean_over_time"],
    }

