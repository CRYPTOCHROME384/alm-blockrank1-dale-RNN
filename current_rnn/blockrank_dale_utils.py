import json
import math
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch


UNKNOWN_LABELS = {"", "unknown", "nan", "none", "null", "na"}


def normalize_block_label(x: Any) -> str:
    if x is None:
        return "unknown"
    s = str(x).strip()
    return s if s != "" else "unknown"


def load_registry_dataframe(registry_dir: str, animal: str) -> Tuple[pd.DataFrame, str]:
    cand1 = os.path.join(str(registry_dir), f"{animal}_registry.csv")
    cand2 = os.path.join(str(registry_dir), "registry.csv")
    path = cand1 if os.path.isfile(cand1) else cand2
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Cannot find registry csv in {registry_dir} (tried {cand1} and {cand2})")
    df = pd.read_csv(path, low_memory=False)
    required = ["unit_key", "global_idx", "array_idx", "npz_path", "cell_sign"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Registry missing required columns {missing}; available columns={list(df.columns)}")
    return df.copy(), path


def subset_and_reindex_registry_dataframe(
    df: pd.DataFrame,
    *,
    max_sessions: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[int, int]]:
    df = df.copy()
    df["unit_key"] = df["unit_key"].astype(str)
    df["global_idx"] = pd.to_numeric(df["global_idx"], errors="raise").astype(int)
    df["array_idx"] = pd.to_numeric(df["array_idx"], errors="raise").astype(int)

    selected_unit_keys = sorted(df["unit_key"].unique().tolist())
    if max_sessions is not None and int(max_sessions) > 0:
        selected_unit_keys = selected_unit_keys[: int(max_sessions)]
        df = df[df["unit_key"].isin(selected_unit_keys)].copy()

    unique_g = sorted(int(x) for x in df["global_idx"].unique().tolist())
    reindex_map = {int(g): int(i) for i, g in enumerate(unique_g)}
    df["global_idx_original"] = df["global_idx"].astype(int)
    df["global_idx"] = df["global_idx"].map(reindex_map).astype(int)
    return df, selected_unit_keys, reindex_map


def build_by_unit_from_registry_dataframe(df: pd.DataFrame) -> Dict[str, List[Tuple[int, int, Dict[str, Any]]]]:
    by_unit: Dict[str, List[Tuple[int, int, Dict[str, Any]]]] = {}
    for row in df.to_dict("records"):
        uk = str(row["unit_key"])
        g = int(row["global_idx"])
        a = int(row["array_idx"])
        by_unit.setdefault(uk, []).append((g, a, dict(row)))
    return by_unit


def infer_block_type_info_from_registry_dataframe(
    df: pd.DataFrame,
    *,
    celltype_mode: str = "broadE_inh_subclass",
    registry_label_cols: Optional[Sequence[str]] = None,
    inh_min_count: int = 10,
    other_inh_label: str = "I_other",
) -> Dict[str, Any]:
    if str(celltype_mode) != "broadE_inh_subclass":
        raise ValueError(f"Unsupported celltype_mode={celltype_mode!r}")
    if registry_label_cols is None:
        registry_label_cols = ["cell_subclass", "cell_cluster", "cell_type", "celltype"]

    df = df.copy()
    gvals = pd.to_numeric(df["global_idx"], errors="raise").astype(int).to_numpy()
    if int(df.shape[0]) == 0:
        raise ValueError("Registry dataframe is empty.")
    if int(gvals.min()) < 0:
        raise ValueError("global_idx must be >= 0.")
    n_obs = int(gvals.max()) + 1

    sign_by_g = [None] * n_obs
    raw_label_by_g = ["unknown"] * n_obs
    label_col_used = None
    for col in registry_label_cols:
        if str(col) not in df.columns:
            continue
        vals = [normalize_block_label(x) for x in df[str(col)].tolist()]
        usable = [v for v in vals if v.strip().lower() not in UNKNOWN_LABELS]
        if len(usable) > 0:
            label_col_used = str(col)
            break

    for row in df.to_dict("records"):
        g = int(row["global_idx"])
        sign = str(row["cell_sign"]).strip().lower()
        if sign_by_g[g] is None:
            sign_by_g[g] = sign
        elif sign_by_g[g] != sign:
            raise ValueError(f"Registry global_idx={g} has inconsistent cell_sign values.")

        if label_col_used is not None:
            lab = normalize_block_label(row.get(label_col_used, None))
            if raw_label_by_g[g] in {"unknown", lab}:
                raw_label_by_g[g] = lab
            elif lab != "unknown":
                raise ValueError(
                    f"Registry global_idx={g} has inconsistent {label_col_used} values: "
                    f"{raw_label_by_g[g]!r} vs {lab!r}"
                )

    inh_raw = [
        raw_label_by_g[g]
        for g in range(n_obs)
        if sign_by_g[g] == "inhibitory" and raw_label_by_g[g].strip().lower() not in UNKNOWN_LABELS
    ]
    inh_raw_counts = Counter(inh_raw)
    reliable_inh = sorted([lab for lab, n in inh_raw_counts.items() if int(n) >= int(inh_min_count)])

    observed_type_labels: List[str] = []
    merged_inh_counts: Dict[str, int] = {}
    for g in range(n_obs):
        sign = sign_by_g[g]
        if sign is None:
            raise ValueError(f"Missing sign for global_idx={g}")
        if sign == "excitatory":
            observed_type_labels.append("E_broad")
            continue

        raw_lab = normalize_block_label(raw_label_by_g[g])
        if raw_lab in reliable_inh:
            observed_type_labels.append(raw_lab)
        else:
            observed_type_labels.append(str(other_inh_label))
            merged_inh_counts[raw_lab] = merged_inh_counts.get(raw_lab, 0) + 1

    type_names = ["E_broad"] + list(reliable_inh)
    if any(x == str(other_inh_label) for x in observed_type_labels):
        type_names.append(str(other_inh_label))

    type_signs = [1.0 if name == "E_broad" else -1.0 for name in type_names]
    type_name_to_index = {name: i for i, name in enumerate(type_names)}
    observed_type_index = np.asarray([type_name_to_index[name] for name in observed_type_labels], dtype=np.int64)
    observed_is_exc = np.asarray([name == "E_broad" for name in observed_type_labels], dtype=np.bool_)
    observed_type_counts = Counter(observed_type_labels)

    notes = []
    if label_col_used is None:
        notes.append("No usable inhibitory subclass column found; all inhibitory neurons were merged into I_other.")
    if len(merged_inh_counts) > 0:
        notes.append(
            "Merged inhibitory labels into I_other because they were missing or below count threshold: "
            + ", ".join([f"{k}={v}" for k, v in sorted(merged_inh_counts.items())])
        )

    return {
        "celltype_mode": str(celltype_mode),
        "registry_label_col_used": label_col_used,
        "inh_min_count": int(inh_min_count),
        "other_inh_label": str(other_inh_label),
        "reliable_inh_labels": list(reliable_inh),
        "merged_inh_counts": dict(sorted(merged_inh_counts.items())),
        "observed_type_labels": np.asarray(observed_type_labels, dtype=object),
        "observed_type_index": observed_type_index,
        "observed_is_exc": observed_is_exc,
        "type_names": list(type_names),
        "type_signs": list(type_signs),
        "observed_type_counts": {k: int(v) for k, v in sorted(observed_type_counts.items())},
        "inh_raw_counts": {k: int(v) for k, v in sorted(inh_raw_counts.items())},
        "notes": notes,
    }


def build_full_block_type_info(
    observed_info: Dict[str, Any],
    *,
    n_exc_virtual: int = 0,
) -> Dict[str, Any]:
    type_names = list(observed_info["type_names"])
    type_signs = list(observed_info["type_signs"])
    type_name_to_index = {name: i for i, name in enumerate(type_names)}
    e_index = int(type_name_to_index["E_broad"])

    obs_index = np.asarray(observed_info["observed_type_index"], dtype=np.int64)
    obs_labels = np.asarray(observed_info["observed_type_labels"], dtype=object)

    n_virtual = int(max(int(n_exc_virtual), 0))
    full_index = np.concatenate([np.full(n_virtual, e_index, dtype=np.int64), obs_index], axis=0)
    full_labels = np.concatenate([np.asarray(["E_broad"] * n_virtual, dtype=object), obs_labels], axis=0)
    full_is_exc = np.asarray([type_names[int(i)] == "E_broad" for i in full_index.tolist()], dtype=np.bool_)
    full_counts = Counter(full_labels.tolist())

    return {
        "type_names": type_names,
        "type_signs": type_signs,
        "type_name_to_index": type_name_to_index,
        "full_type_index": full_index,
        "full_type_labels": full_labels,
        "full_is_exc": full_is_exc,
        "full_type_counts": {k: int(v) for k, v in sorted(full_counts.items())},
        "observed_type_counts": dict(observed_info["observed_type_counts"]),
        "n_exc_virtual": int(n_virtual),
        "block_list": [f"{a}<-{b}" for a in type_names for b in type_names],
        "presyn_count_by_type": {k: int(v) for k, v in sorted(full_counts.items())},
    }


def local_type_masks_from_idx_net(
    idx_net: torch.Tensor,
    *,
    full_type_index: np.ndarray,
    type_names: Sequence[str],
) -> Dict[str, torch.Tensor]:
    idx_np = idx_net.detach().cpu().numpy().astype(np.int64)
    local_type_index = np.asarray(full_type_index, dtype=np.int64)[idx_np]
    masks: Dict[str, torch.Tensor] = {}
    for t, name in enumerate(type_names):
        masks[str(name)] = torch.as_tensor(local_type_index == int(t), dtype=torch.bool, device=idx_net.device)
    return masks


def local_sign_masks_from_idx_net(
    idx_net: torch.Tensor,
    *,
    full_is_exc: np.ndarray,
) -> Dict[str, torch.Tensor]:
    idx_np = idx_net.detach().cpu().numpy().astype(np.int64)
    local_is_exc = np.asarray(full_is_exc, dtype=np.bool_)[idx_np]
    return {
        "excitatory": torch.as_tensor(local_is_exc, dtype=torch.bool, device=idx_net.device),
        "inhibitory": torch.as_tensor(~local_is_exc, dtype=torch.bool, device=idx_net.device),
    }


def build_epoch_masks(meta: Dict[str, Any], T: int, sample_ignore_ms: float, resp_sec: float) -> Dict[str, np.ndarray]:
    fps = float(meta["fps"])
    event_frames = meta.get("event_frames", {}) or {}
    S = int(event_frames.get("S", 0))
    D = int(event_frames.get("D", S))
    R = int(event_frames.get("R", D))
    ignore_frames = int(round(float(sample_ignore_ms) * fps / 1000.0))
    resp_frames = int(round(float(resp_sec) * fps))
    idx = np.arange(int(T), dtype=np.int64)

    sample_start = max(0, S + ignore_frames)
    sample_mask = (idx >= sample_start) & (idx < D)
    delay_mask = (idx >= D) & (idx < R)
    response_mask = (idx >= R) & (idx < min(int(T), R + resp_frames))
    loss_mask = sample_mask | delay_mask | response_mask
    return {
        "sample": sample_mask,
        "delay": delay_mask,
        "response": response_mask,
        "loss": loss_mask,
    }


def _safe_ratio(num: float, den: float) -> float:
    if not math.isfinite(float(num)) or not math.isfinite(float(den)):
        return float("nan")
    if abs(float(den)) <= 1e-12:
        return float("nan")
    return float(num) / float(den)


def compute_delay_ratio_metrics(
    *,
    target: torch.Tensor,
    pred: torch.Tensor,
    epoch_masks: Dict[str, np.ndarray],
    local_masks: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    if tuple(target.shape) != tuple(pred.shape):
        raise ValueError(f"target/pred shape mismatch: {tuple(target.shape)} vs {tuple(pred.shape)}")
    if target.dim() != 3 or int(target.shape[0]) < 2:
        raise ValueError(f"Expected [C>=2, T, K] tensors, got {tuple(target.shape)}")

    def _mean_curve(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x[0] + x[1])

    def _metric_for_mask(mask_np: np.ndarray, x: torch.Tensor, stat: str) -> torch.Tensor:
        if int(np.asarray(mask_np, dtype=np.bool_).sum()) <= 0:
            return torch.full((x.shape[-1],), float("nan"), device=x.device, dtype=x.dtype)
        m = torch.as_tensor(np.asarray(mask_np, dtype=np.bool_), device=x.device)
        xs = x[m, :]
        if stat == "std":
            return xs.std(dim=0, unbiased=False)
        if stat == "mean_abs":
            return xs.abs().mean(dim=0)
        raise ValueError(f"Unsupported stat={stat!r}")

    mean_target = _mean_curve(target)
    mean_pred = _mean_curve(pred)
    delay_std_gt = _metric_for_mask(epoch_masks["delay"], mean_target, "std")
    delay_std_pr = _metric_for_mask(epoch_masks["delay"], mean_pred, "std")
    sample_std_gt = _metric_for_mask(epoch_masks["sample"], mean_target, "std")
    sample_std_pr = _metric_for_mask(epoch_masks["sample"], mean_pred, "std")
    resp_std_gt = _metric_for_mask(epoch_masks["response"], mean_target, "std")
    resp_std_pr = _metric_for_mask(epoch_masks["response"], mean_pred, "std")
    delay_mean_abs_gt = _metric_for_mask(epoch_masks["delay"], mean_target, "mean_abs")
    delay_mean_abs_pr = _metric_for_mask(epoch_masks["delay"], mean_pred, "mean_abs")

    delay_mask = torch.as_tensor(np.asarray(epoch_masks["delay"], dtype=np.bool_), device=target.device)
    if int(delay_mask.sum().item()) > 0:
        lr_gt = (target[1, delay_mask, :].mean(dim=0) - target[0, delay_mask, :].mean(dim=0)).abs()
        lr_pr = (pred[1, delay_mask, :].mean(dim=0) - pred[0, delay_mask, :].mean(dim=0)).abs()
    else:
        lr_gt = torch.full((target.shape[-1],), float("nan"), device=target.device, dtype=target.dtype)
        lr_pr = torch.full((target.shape[-1],), float("nan"), device=target.device, dtype=target.dtype)

    out: Dict[str, Dict[str, float]] = {}
    for group_name, group_mask in local_masks.items():
        gm = group_mask.bool()
        if int(gm.sum().item()) <= 0:
            continue

        def _mean_valid(v: torch.Tensor) -> float:
            vals = v[gm].detach().cpu().numpy().astype(float)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size > 0 else float("nan")

        delay_std_gt_mean = _mean_valid(delay_std_gt)
        delay_std_pr_mean = _mean_valid(delay_std_pr)
        delay_mean_abs_gt_mean = _mean_valid(delay_mean_abs_gt)
        delay_mean_abs_pr_mean = _mean_valid(delay_mean_abs_pr)
        sample_std_gt_mean = _mean_valid(sample_std_gt)
        sample_std_pr_mean = _mean_valid(sample_std_pr)
        resp_std_gt_mean = _mean_valid(resp_std_gt)
        resp_std_pr_mean = _mean_valid(resp_std_pr)
        lr_gt_mean = _mean_valid(lr_gt)
        lr_pr_mean = _mean_valid(lr_pr)

        out[str(group_name)] = {
            "n_cells": int(gm.sum().item()),
            "delay_std_ratio": _safe_ratio(delay_std_pr_mean, delay_std_gt_mean),
            "delay_lr_sep_ratio": _safe_ratio(lr_pr_mean, lr_gt_mean),
            "delay_mean_abs_ratio": _safe_ratio(delay_mean_abs_pr_mean, delay_mean_abs_gt_mean),
            "sample_std_ratio": _safe_ratio(sample_std_pr_mean, sample_std_gt_mean),
            "resp_std_ratio": _safe_ratio(resp_std_pr_mean, resp_std_gt_mean),
            "delay_std_target": float(delay_std_gt_mean),
            "delay_std_pred": float(delay_std_pr_mean),
            "delay_mean_abs_target": float(delay_mean_abs_gt_mean),
            "delay_mean_abs_pred": float(delay_mean_abs_pr_mean),
            "sample_std_target": float(sample_std_gt_mean),
            "sample_std_pred": float(sample_std_pr_mean),
            "resp_std_target": float(resp_std_gt_mean),
            "resp_std_pred": float(resp_std_pr_mean),
            "delay_lr_sep_target": float(lr_gt_mean),
            "delay_lr_sep_pred": float(lr_pr_mean),
        }
    return out


def save_blockrank_diagnostics(
    *,
    model,
    out_dir: str,
    stem: str,
    extra: Optional[Dict[str, Any]] = None,
    include_dense_summary: bool = True,
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload: Dict[str, Any] = {
        "dale_violation_count": int(model.dale_violation_count()),
        "block_parameter_summary": model.block_parameter_summary(),
        "numerical_block_ranks": model.numerical_block_ranks(),
    }
    if bool(include_dense_summary):
        J = model.materialize_J_for_debug().detach().cpu()
        payload["reconstructed_J_summary"] = {
            "shape": list(J.shape),
            "mean": float(J.mean().item()),
            "std": float(J.std(unbiased=False).item()),
            "min": float(J.min().item()),
            "max": float(J.max().item()),
            "frob": float(torch.linalg.norm(J).item()),
        }
    if isinstance(extra, dict):
        payload.update(extra)

    path = os.path.join(out_dir, f"{stem}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path
