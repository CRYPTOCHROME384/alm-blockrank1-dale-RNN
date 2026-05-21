from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as tch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from blockrank_dale_utils import (
    build_by_unit_from_registry_dataframe,
    build_epoch_masks,
    build_full_block_type_info,
    compute_delay_ratio_metrics,
    infer_block_type_info_from_registry_dataframe,
    load_registry_dataframe,
    subset_and_reindex_registry_dataframe,
    save_blockrank_diagnostics,
)
from delay_dynamics import (
    DELAY_SHAPE_LOSS_RAW,
    DELAY_SHAPE_LOSS_REL_GLOBAL,
    DELAY_SHAPE_LOSS_REL_GROUP,
    compute_delay_component_errors,
    compute_delay_shape_loss_bundle,
    compute_delay_statistics,
    compute_relative_ac_loss,
    delay_centered_shape_loss,
    extract_delay_segment,
    make_constant_delay_baseline,
    summarize_condition_neuron_metrics,
)
from eval_current_alm import compute_time_mask_tsec_zero_at_R, plot_psth_comparison_R0_with_events
from losses import LossAverageTrials
from models import CellTypeBlockRank1DaleCurrentRNN
from training_blockrank1_dale import _configure_runtime_device, _select_psth_target
from training_current import _preload_units_from_registry


EPOCH_NAMES = ("sample", "delay", "response")
DEFAULT_LAMBDA_CANDIDATES = (0.03, 0.1, 0.3, 1.0, 3.0)
DEFAULT_FINETUNE_EPOCHS = (500, 2000)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _write_json(obj: Dict[str, Any], path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def _save_df(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _default_out_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(ROOT_DIR, "results_current", f"blockrank1_dale_preflight_{ts}")


def _float_item(x: Any) -> float:
    if isinstance(x, tch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)


def _mean_or_nan(vals: Sequence[float]) -> float:
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size > 0 else float("nan")


def _safe_ratio(num: float, den: float) -> float:
    num_f = float(num)
    den_f = float(den)
    if not math.isfinite(num_f) or not math.isfinite(den_f) or abs(den_f) <= 1e-12:
        return float("nan")
    return num_f / den_f


def _parse_float_csv(text: str) -> List[float]:
    out: List[float] = []
    for piece in str(text).split(","):
        s = piece.strip()
        if s == "":
            continue
        out.append(float(s))
    return out


def _parse_int_csv(text: str) -> List[int]:
    out: List[int] = []
    for piece in str(text).split(","):
        s = piece.strip()
        if s == "":
            continue
        out.append(int(s))
    return out


def _seed_all(seed: int, device: tch.device) -> None:
    np.random.seed(int(seed))
    tch.manual_seed(int(seed))
    if device.type == "cuda":
        tch.cuda.manual_seed_all(int(seed))


def _load_checkpoint_state(path: str, device: tch.device) -> Dict[str, Any]:
    ckpt = tch.load(str(path), map_location=device)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise TypeError(f"Unexpected checkpoint payload type at {path}: {type(ckpt)!r}")


def _build_model_from_cfg(
    *,
    cfg: Dict[str, Any],
    n_total: int,
    d_in: int,
    neuron_type_index: tch.Tensor,
    type_names: List[str],
    type_signs: List[float],
    device: tch.device,
) -> CellTypeBlockRank1DaleCurrentRNN:
    return CellTypeBlockRank1DaleCurrentRNN(
        N=int(n_total),
        D_in=int(d_in),
        neuron_type_index=neuron_type_index,
        type_names=list(type_names),
        type_signs=list(type_signs),
        dt=float(cfg.get("dt", 0.03436)),
        tau=float(cfg.get("tau", 0.01)),
        substeps=int(cfg.get("substeps", 4)),
        nonlinearity=str(cfg.get("nonlinearity", "tanh")),
        block_rank=int(cfg.get("block_rank", 1)),
        factor_nonlinearity=str(cfg.get("factor_nonlinearity", "softplus")),
        A_nonlinearity=str(cfg.get("A_nonlinearity", "softplus")),
        normalize_uv=bool(cfg.get("normalize_uv", True)),
        eps=float(cfg.get("eps", 1e-8)),
        init_A=float(cfg.get("init_A", 0.1)),
        init_factor_scale=float(cfg.get("init_factor_scale", 0.02)),
        device=device,
    ).to(device)


def _load_units_and_block_info(
    cfg: Dict[str, Any],
    device: tch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame]:
    registry_dir = str(cfg["registry_dir"])
    animal = str(cfg["animal"])
    registry_df_full, _ = load_registry_dataframe(registry_dir=registry_dir, animal=animal)
    max_sessions_raw = cfg.get("max_sessions", None)
    max_sessions = None if max_sessions_raw in (None, "", 0) else int(max_sessions_raw)
    registry_df, _, _ = subset_and_reindex_registry_dataframe(registry_df_full, max_sessions=max_sessions)
    by_unit = build_by_unit_from_registry_dataframe(registry_df)
    units, shared = _preload_units_from_registry(
        by_unit=by_unit,
        n_exc_virtual=int(cfg.get("n_exc_virtual", 0)),
        device=device,
        cond_filter=cfg.get("cond_filter", None),
        max_time=cfg.get("max_time", None),
        psth_bin_ms=float(cfg.get("psth_bin_ms", 200.0)),
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
        use_trials=bool(cfg.get("use_trials", True)),
        strict_trials_align=bool(cfg.get("strict_trials_align", True)),
        trials_root=cfg.get("trials_root", None),
        trials_path_mode=str(cfg.get("trials_path_mode", "auto_from_stage1")),
        trial_keys=cfg.get("trial_keys", None),
        debug_trials_align=bool(cfg.get("debug_trials_align", False)),
        phase3_precompute_var_real=False,
        var_unbiased=False,
        min_trials_for_var_real=3,
    )
    if len(units) == 0:
        raise RuntimeError("No units loaded for preflight.")

    block_info_obs = infer_block_type_info_from_registry_dataframe(
        registry_df,
        celltype_mode=str(cfg.get("celltype_mode", "broadE_inh_subclass")),
        registry_label_cols=cfg.get("block_registry_label_cols", None),
        inh_min_count=int(cfg.get("block_celltype_min_count", 10)),
        other_inh_label=str(cfg.get("other_inh_label", "I_other")),
    )
    block_info_full = build_full_block_type_info(block_info_obs, n_exc_virtual=int(cfg.get("n_exc_virtual", 0)))
    return units, shared, block_info_obs, block_info_full, registry_df


def _masked_mse(loss_fn: LossAverageTrials, target: tch.Tensor, pred: tch.Tensor, mask_np: np.ndarray) -> tch.Tensor:
    mask_t = tch.as_tensor(np.asarray(mask_np, dtype=np.bool_), device=target.device)
    if int(mask_t.sum().item()) <= 0:
        return target.new_full((), float("nan"))
    return loss_fn(target[:, mask_t, :], pred[:, mask_t, :])


def _epoch_losses(
    *,
    loss_fn: LossAverageTrials,
    target: tch.Tensor,
    pred: tch.Tensor,
    epoch_masks: Dict[str, np.ndarray],
) -> Dict[str, tch.Tensor]:
    return {name: _masked_mse(loss_fn, target, pred, epoch_masks[name]) for name in EPOCH_NAMES}


def _epoch_normalized_losses(
    *,
    loss_fn: LossAverageTrials,
    target: tch.Tensor,
    pred: tch.Tensor,
    epoch_masks: Dict[str, np.ndarray],
    eps: float = 1e-8,
) -> Dict[str, tch.Tensor]:
    out: Dict[str, tch.Tensor] = {}
    for name in EPOCH_NAMES:
        mask_t = tch.as_tensor(np.asarray(epoch_masks[name], dtype=np.bool_), device=target.device)
        if int(mask_t.sum().item()) <= 0:
            out[name] = target.new_full((), float("nan"))
            continue
        loss = loss_fn(target[:, mask_t, :], pred[:, mask_t, :])
        scale = target[:, mask_t, :].pow(2).mean().detach().clamp_min(float(eps))
        out[name] = loss / scale
    return out


def _weighted_mean_loss(losses: Dict[str, tch.Tensor], weights: Dict[str, float]) -> tch.Tensor:
    terms: List[tch.Tensor] = []
    active_weights: List[float] = []
    fallback = None
    for name in EPOCH_NAMES:
        val = losses[name]
        if fallback is None and bool(tch.isfinite(val).item()):
            fallback = val
        w = float(weights.get(name, 0.0))
        if w > 0.0 and bool(tch.isfinite(val).item()):
            terms.append(val * w)
            active_weights.append(w)
    if len(terms) > 0:
        return tch.stack(terms).sum() / float(sum(active_weights))
    if fallback is not None:
        return fallback
    return next(iter(losses.values()))


def _cfg_loss_epoch_weights(cfg: Dict[str, Any]) -> Dict[str, float]:
    src = cfg.get("loss_epoch_weights", {"sample": 1.0, "delay": 1.0, "response": 1.0})
    return {str(k): float(v) for k, v in dict(src).items()}


def _select_main_loss(
    *,
    cfg: Dict[str, Any],
    raw_epoch_losses: Dict[str, tch.Tensor],
    norm_epoch_losses: Dict[str, tch.Tensor],
) -> tch.Tensor:
    mode = str(cfg.get("loss_mode", "mean")).strip().lower()
    weights = _cfg_loss_epoch_weights(cfg)
    if mode == "epoch_weighted_mean":
        return _weighted_mean_loss(raw_epoch_losses, weights)
    return _weighted_mean_loss(raw_epoch_losses, weights)


def _local_type_arrays(batch: Dict[str, Any], block_info_full: Dict[str, Any]) -> Dict[str, Any]:
    idx_np = batch["idx_net"].detach().cpu().numpy().astype(np.int64)
    full_type_index = np.asarray(block_info_full["full_type_index"], dtype=np.int64)
    full_is_exc = np.asarray(block_info_full["full_is_exc"], dtype=np.bool_)
    local_type_index = full_type_index[idx_np]
    local_is_exc = full_is_exc[idx_np]
    local_type_names = [str(block_info_full["type_names"][int(i)]) for i in local_type_index.tolist()]
    return {
        "local_type_index": local_type_index,
        "local_is_exc": local_is_exc,
        "local_type_names": local_type_names,
    }


def _local_delay_group_args(
    *,
    batch: Dict[str, Any],
    block_info_full: Dict[str, Any],
    group_mode: str,
) -> Dict[str, Any]:
    mode = str(group_mode or "all").strip().lower()
    if mode == "all":
        return {"group_mode": "all", "group_index": None, "group_names": None}
    if mode != "celltype":
        raise ValueError(f"Unsupported delay_shape_group_mode={group_mode!r}")
    local = _local_type_arrays(batch, block_info_full)
    return {
        "group_mode": "celltype",
        "group_index": local["local_type_index"].tolist(),
        "group_names": list(block_info_full["type_names"]),
    }


def _delay_lr_per_neuron(x: tch.Tensor, delay_mask: np.ndarray) -> np.ndarray:
    mask_t = tch.as_tensor(np.asarray(delay_mask, dtype=np.bool_), device=x.device)
    if int(x.shape[0]) < 2 or int(mask_t.sum().item()) <= 0:
        return np.full((int(x.shape[-1]),), float("nan"), dtype=float)
    lr = (x[1, mask_t, :].mean(dim=0) - x[0, mask_t, :].mean(dim=0)).abs()
    return lr.detach().cpu().numpy().astype(float)


def _summarize_groups(
    *,
    df: pd.DataFrame,
    metrics: Sequence[str],
    group_specs: Sequence[Tuple[str, Optional[str]]],
    ratio_specs: Optional[Sequence[Tuple[str, str, str]]] = None,
) -> pd.DataFrame:
    ratio_specs = list(ratio_specs or [])
    rows: List[Dict[str, Any]] = []
    for family, col in group_specs:
        if col is None:
            row = {
                "group_family": str(family),
                "group_name": "all",
                "n_rows": int(df.shape[0]),
                **{m: _mean_or_nan(df[m].tolist()) for m in metrics},
            }
            for out_name, num_col, den_col in ratio_specs:
                row[str(out_name)] = _safe_ratio(_mean_or_nan(df[str(num_col)].tolist()), _mean_or_nan(df[str(den_col)].tolist()))
            rows.append(row)
            continue
        for group_name, sub in df.groupby(col, dropna=False):
            row = {
                "group_family": str(family),
                "group_name": str(group_name),
                "n_rows": int(sub.shape[0]),
                **{m: _mean_or_nan(sub[m].tolist()) for m in metrics},
            }
            for out_name, num_col, den_col in ratio_specs:
                row[str(out_name)] = _safe_ratio(_mean_or_nan(sub[str(num_col)].tolist()), _mean_or_nan(sub[str(den_col)].tolist()))
            rows.append(row)
    return pd.DataFrame(rows)


def _plot_top_target_neurons(
    *,
    records: List[Dict[str, Any]],
    units_by_key: Dict[str, Dict[str, Any]],
    out_png: str,
    top_n: int,
    sample_ignore_ms: float,
    resp_sec: float,
) -> Optional[str]:
    if len(records) == 0 or int(top_n) <= 0:
        return None
    sorted_records = sorted(records, key=lambda r: float(r["target_delay_std_mean_over_conditions"]), reverse=True)[: int(top_n)]
    n = len(sorted_records)
    ncols = min(3, n)
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, rec in zip(axes.flat, sorted_records):
        batch = units_by_key[str(rec["unit_key"])]
        target = tch.nan_to_num(_select_psth_target(batch, str(batch["meta"]["_psth_target_source"])), nan=0.0, posinf=0.0, neginf=0.0)
        epoch_masks = build_epoch_masks(
            batch["meta"],
            T=int(target.shape[1]),
            sample_ignore_ms=float(sample_ignore_ms),
            resp_sec=float(resp_sec),
        )
        delay_mask = np.asarray(epoch_masks["delay"], dtype=np.bool_)
        seg = extract_delay_segment(target, delay_mask).detach().cpu().numpy().astype(float)
        fps = float(batch["meta"]["fps"])
        t = np.arange(seg.shape[1], dtype=float) / fps
        local_i = int(rec["local_neuron_index"])
        for ci, cond_name in enumerate(batch["meta"]["cond_names"]):
            ax.plot(t, seg[ci, :, local_i], lw=1.8, label=str(cond_name))
        ax.set_title(
            f"{rec['unit_key']} n={local_i} {rec['type_name']} std={rec['target_delay_std_mean_over_conditions']:.4f}",
            fontsize=10,
        )
        ax.set_xlabel("Delay Time (s)")
        ax.set_ylabel("Target PSTH")
        ax.axhline(0.0, color="black", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8, frameon=False)
        ax.axis("on")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def _pick_probe_batch(
    units: List[Dict[str, Any]],
    block_info_full: Dict[str, Any],
    *,
    sample_ignore_ms: float,
    resp_sec: float,
    psth_target_source: str,
) -> Dict[str, Any]:
    best = None
    best_score = -float("inf")
    for batch in units:
        target = tch.nan_to_num(_select_psth_target(batch, str(psth_target_source)), nan=0.0, posinf=0.0, neginf=0.0)
        epoch_masks = build_epoch_masks(
            batch["meta"],
            T=int(target.shape[1]),
            sample_ignore_ms=float(sample_ignore_ms),
            resp_sec=float(resp_sec),
        )
        delay_mask = np.asarray(epoch_masks["delay"], dtype=np.bool_)
        stats = compute_delay_statistics(target, delay_mask)
        mean_std = stats["std"].mean(dim=0).detach().cpu().numpy().astype(float)
        local_info = _local_type_arrays(batch, block_info_full)
        inh_count = int((~local_info["local_is_exc"]).sum())
        score = float(np.nanmean(mean_std))
        if inh_count > 0:
            score += 0.01
        if score > best_score:
            best_score = score
            best = batch
    if best is None:
        raise RuntimeError("Could not pick probe batch.")
    return best


def _grad_group_norms(model: CellTypeBlockRank1DaleCurrentRNN) -> Dict[str, Any]:
    sums = {"A_raw": 0.0, "u_raw": 0.0, "v_raw": 0.0, "W_in": 0.0, "bias": 0.0, "other": 0.0}
    nonzero = {k: 0 for k in sums}
    has_nan = False
    max_abs = 0.0
    total_sq = 0.0
    x0_sq = None
    x0_nonzero = None
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        if not tch.isfinite(g).all():
            has_nan = True
        sq = float(g.pow(2).sum().item())
        total_sq += sq
        max_abs = max(max_abs, float(g.abs().max().item()))
        bucket = "other"
        if name.endswith("A_raw"):
            bucket = "A_raw"
        elif name.endswith("u_raw"):
            bucket = "u_raw"
        elif name.endswith("v_raw"):
            bucket = "v_raw"
        elif name == "W_in":
            bucket = "W_in"
        elif name == "b":
            bucket = "bias"
        elif name in {"x0", "h0"}:
            bucket = "other"
            x0_sq = sq
            x0_nonzero = int((g.abs() > 0).sum().item())
        sums[bucket] += sq
        nonzero[bucket] += int((g.abs() > 0).sum().item())

    out = {
        "A_raw_grad_norm": float(math.sqrt(max(sums["A_raw"], 0.0))),
        "u_raw_grad_norm": float(math.sqrt(max(sums["u_raw"], 0.0))),
        "v_raw_grad_norm": float(math.sqrt(max(sums["v_raw"], 0.0))),
        "W_in_grad_norm": float(math.sqrt(max(sums["W_in"], 0.0))),
        "bias_grad_norm": float(math.sqrt(max(sums["bias"], 0.0))),
        "other_grad_norm": float(math.sqrt(max(sums["other"], 0.0))),
        "total_grad_norm": float(math.sqrt(max(total_sq, 0.0))),
        "A_raw_nonzero": int(nonzero["A_raw"]),
        "u_raw_nonzero": int(nonzero["u_raw"]),
        "v_raw_nonzero": int(nonzero["v_raw"]),
        "W_in_nonzero": int(nonzero["W_in"]),
        "bias_nonzero": int(nonzero["bias"]),
        "other_nonzero": int(nonzero["other"]),
        "x0_grad_norm": None if x0_sq is None else float(math.sqrt(max(x0_sq, 0.0))),
        "x0_nonzero": x0_nonzero,
        "has_nonfinite_grad": bool(has_nan),
        "grad_max_abs": float(max_abs),
    }
    return out


def _build_target_prediction_tables(
    *,
    batch: Dict[str, Any],
    target: tch.Tensor,
    pred: tch.Tensor,
    block_info_full: Dict[str, Any],
    delay_shape_eps: float,
    delay_shape_min_scale: float,
    delay_shape_group_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    epoch_masks = build_epoch_masks(
        batch["meta"],
        T=int(target.shape[1]),
        sample_ignore_ms=float(batch["meta"].get("_sample_ignore_ms", 50.0)),
        resp_sec=float(batch["meta"].get("_resp_sec", 2.0)),
    )
    delay_mask = np.asarray(epoch_masks["delay"], dtype=np.bool_)
    const_pred = make_constant_delay_baseline(target, delay_mask)

    local_info = _local_type_arrays(batch, block_info_full)
    cond_names = [str(x) for x in batch["meta"]["cond_names"]]
    group_args = _local_delay_group_args(batch=batch, block_info_full=block_info_full, group_mode=delay_shape_group_mode)

    model_metrics = summarize_condition_neuron_metrics(
        target=target,
        pred=pred,
        delay_mask=delay_mask,
        group_mode=str(group_args["group_mode"]),
        group_index=group_args["group_index"],
        group_names=group_args["group_names"],
        eps=float(delay_shape_eps),
        min_scale=float(delay_shape_min_scale),
    )
    const_metrics = summarize_condition_neuron_metrics(
        target=target,
        pred=const_pred,
        delay_mask=delay_mask,
        group_mode=str(group_args["group_mode"]),
        group_index=group_args["group_index"],
        group_names=group_args["group_names"],
        eps=float(delay_shape_eps),
        min_scale=float(delay_shape_min_scale),
    )
    target_stats = compute_delay_statistics(target, delay_mask)
    lr_target = _delay_lr_per_neuron(target, delay_mask)
    lr_model = _delay_lr_per_neuron(pred, delay_mask)
    lr_const = _delay_lr_per_neuron(const_pred, delay_mask)

    target_rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []
    top_rows: List[Dict[str, Any]] = []
    lr_rows: List[Dict[str, Any]] = []

    target_std_ck = target_stats["std"].detach().cpu().numpy().astype(float)
    target_deriv_ck = target_stats["derivative_rms"].detach().cpu().numpy().astype(float)
    target_energy_ck = target_stats["centered_energy"].detach().cpu().numpy().astype(float)

    for local_k in range(int(target.shape[-1])):
        type_name = str(local_info["local_type_names"][local_k])
        sign_group = "E" if bool(local_info["local_is_exc"][local_k]) else "I"
        per_cond_std = target_std_ck[:, local_k]
        top_rows.append(
            {
                "unit_key": str(batch["unit_key"]),
                "local_neuron_index": int(local_k),
                "type_name": type_name,
                "sign_group": sign_group,
                "target_delay_std_mean_over_conditions": _mean_or_nan(per_cond_std.tolist()),
                "delay_lr_sep_target": float(lr_target[local_k]),
            }
        )
        lr_rows.append(
            {
                "unit_key": str(batch["unit_key"]),
                "local_neuron_index": int(local_k),
                "type_name": type_name,
                "sign_group": sign_group,
                "delay_lr_sep_target": float(lr_target[local_k]),
                "delay_lr_sep_pred": float(lr_model[local_k]),
                "delay_lr_sep_ratio": _safe_ratio(float(lr_model[local_k]), float(lr_target[local_k])),
                "constant_delay_lr_sep_pred": float(lr_const[local_k]),
                "constant_delay_lr_sep_ratio": _safe_ratio(float(lr_const[local_k]), float(lr_target[local_k])),
            }
        )
        for ci, cond_name in enumerate(cond_names):
            target_rows.append(
                {
                    "unit_key": str(batch["unit_key"]),
                    "cond_name": str(cond_name),
                    "local_neuron_index": int(local_k),
                    "type_name": type_name,
                    "sign_group": sign_group,
                    "target_delay_std": float(target_std_ck[ci, local_k]),
                    "target_delay_derivative_rms": float(target_deriv_ck[ci, local_k]),
                    "target_delay_centered_energy": float(target_energy_ck[ci, local_k]),
                }
            )
            prediction_rows.append(
                {
                    "unit_key": str(batch["unit_key"]),
                    "cond_name": str(cond_name),
                    "local_neuron_index": int(local_k),
                    "type_name": type_name,
                    "sign_group": sign_group,
                    "target_delay_std": float(model_metrics["target_delay_std"][ci, local_k].detach().cpu().item()),
                    "pred_delay_std": float(model_metrics["pred_delay_std"][ci, local_k].detach().cpu().item()),
                    "delay_std_ratio": float(model_metrics["delay_std_ratio"][ci, local_k].detach().cpu().item()),
                    "delay_mse": float(model_metrics["delay_mse"][ci, local_k].detach().cpu().item()),
                    "delay_shape_raw": float(model_metrics["delay_shape_loss"][ci, local_k].detach().cpu().item()),
                    "dc_error": float(model_metrics["dc_error"][ci, local_k].detach().cpu().item()),
                    "ac_error": float(model_metrics["ac_error"][ci, local_k].detach().cpu().item()),
                    "pred_delay_derivative_rms": float(model_metrics["pred_delay_derivative_rms"][ci, local_k].detach().cpu().item()),
                    "target_delay_derivative_rms": float(model_metrics["target_delay_derivative_rms"][ci, local_k].detach().cpu().item()),
                    "pred_delay_centered_energy": float(model_metrics["pred_delay_centered_energy"][ci, local_k].detach().cpu().item()),
                    "target_delay_centered_energy": float(model_metrics["target_delay_centered_energy"][ci, local_k].detach().cpu().item()),
                    "constant_delay_std": float(const_metrics["pred_delay_std"][ci, local_k].detach().cpu().item()),
                    "constant_delay_std_ratio": float(const_metrics["delay_std_ratio"][ci, local_k].detach().cpu().item()),
                    "constant_delay_mse": float(const_metrics["delay_mse"][ci, local_k].detach().cpu().item()),
                    "constant_delay_shape_raw": float(const_metrics["delay_shape_loss"][ci, local_k].detach().cpu().item()),
                    "constant_dc_error": float(const_metrics["dc_error"][ci, local_k].detach().cpu().item()),
                    "constant_ac_error": float(const_metrics["ac_error"][ci, local_k].detach().cpu().item()),
                }
            )
    return target_rows, prediction_rows, top_rows, lr_rows


def _evaluate_model_on_units(
    *,
    model: CellTypeBlockRank1DaleCurrentRNN,
    units: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    block_info_full: Dict[str, Any],
    out_dir: Optional[str],
    model_path: Optional[str],
    label: str,
) -> Dict[str, Any]:
    model.eval()
    loss_mask_sum = 0.0
    loss_mask_count = 0
    epoch_sq = {name: 0.0 for name in EPOCH_NAMES}
    epoch_count = {name: 0 for name in EPOCH_NAMES}
    delay_raw_sq_sum = 0.0
    delay_target_energy_sum = 0.0
    delay_dc_sum = 0.0
    delay_dc_count = 0
    group_num_sum: Dict[str, float] = {}
    group_den_sum: Dict[str, float] = {}
    const_group_num_sum: Dict[str, float] = {}
    rel_group_warnings: List[str] = []
    rel_const_group_warnings: List[str] = []

    target_rows_all: List[Dict[str, Any]] = []
    pred_rows_all: List[Dict[str, Any]] = []
    top_rows_all: List[Dict[str, Any]] = []
    lr_rows_all: List[Dict[str, Any]] = []

    with tch.no_grad():
        for batch in units:
            target = tch.nan_to_num(_select_psth_target(batch, str(cfg.get("psth_target_source", "trials_mean"))), nan=0.0, posinf=0.0, neginf=0.0)
            pred_full = model(batch["u"], h0=None, noise_std=0.0, return_rate=True)["rate"]
            pred = pred_full.index_select(dim=2, index=batch["idx_net"])
            epoch_masks = build_epoch_masks(
                batch["meta"],
                T=int(target.shape[1]),
                sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
                resp_sec=float(cfg.get("resp_sec", 2.0)),
            )
            delay_mask = np.asarray(epoch_masks["delay"], dtype=np.bool_)
            const_pred = make_constant_delay_baseline(target, delay_mask)

            for name in ["loss"] + list(EPOCH_NAMES):
                mask_t = tch.as_tensor(np.asarray(epoch_masks[name], dtype=np.bool_), device=target.device)
                if int(mask_t.sum().item()) <= 0:
                    continue
                diff_sq = (pred[:, mask_t, :] - target[:, mask_t, :]).pow(2)
                sq_sum = float(diff_sq.sum().detach().cpu().item())
                count = int(diff_sq.numel())
                if name == "loss":
                    loss_mask_sum += sq_sum
                    loss_mask_count += count
                else:
                    epoch_sq[name] += sq_sum
                    epoch_count[name] += count

            delay_errs = compute_delay_component_errors(target, pred, delay_mask)
            delay_target_stats = compute_delay_statistics(target, delay_mask)
            delay_raw_sq_sum += float(delay_errs["shape_loss"].sum().detach().cpu().item()) * int(delay_errs["shape_loss"].shape[0])
            delay_target_energy_sum += float(delay_target_stats["centered_energy"].sum().detach().cpu().item()) * int(delay_target_stats["centered_energy"].shape[0])
            delay_dc_sum += float(delay_errs["dc_error"].sum().detach().cpu().item())
            delay_dc_count += int(delay_errs["dc_error"].numel())

            group_args = _local_delay_group_args(
                batch=batch,
                block_info_full=block_info_full,
                group_mode=str(cfg.get("delay_shape_group_mode", "celltype")),
            )
            rel_bundle = compute_delay_shape_loss_bundle(
                target=target,
                pred=pred,
                delay_mask=delay_mask,
                loss_type=DELAY_SHAPE_LOSS_REL_GROUP,
                group_mode=str(group_args["group_mode"]),
                group_index=group_args["group_index"],
                group_names=group_args["group_names"],
                eps=float(cfg.get("delay_shape_eps", 1e-8)),
                min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
            )
            const_rel_bundle = compute_delay_shape_loss_bundle(
                target=target,
                pred=const_pred,
                delay_mask=delay_mask,
                loss_type=DELAY_SHAPE_LOSS_REL_GROUP,
                group_mode=str(group_args["group_mode"]),
                group_index=group_args["group_index"],
                group_names=group_args["group_names"],
                eps=float(cfg.get("delay_shape_eps", 1e-8)),
                min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
            )
            rel_group_warnings.extend([str(x) for x in rel_bundle["details"].get("warnings", [])])
            rel_const_group_warnings.extend([str(x) for x in const_rel_bundle["details"].get("warnings", [])])
            for row in rel_bundle["details"].get("groups", []):
                name = str(row["group_name"])
                group_num_sum[name] = group_num_sum.get(name, 0.0) + float(row["numerator_mean"]) * float(row["n_points"])
                group_den_sum[name] = group_den_sum.get(name, 0.0) + float(row["target_ac_energy"]) * float(row["n_points"])
            for row in const_rel_bundle["details"].get("groups", []):
                name = str(row["group_name"])
                const_group_num_sum[name] = const_group_num_sum.get(name, 0.0) + float(row["numerator_mean"]) * float(row["n_points"])

            target_rows, pred_rows, top_rows, lr_rows = _build_target_prediction_tables(
                batch=batch,
                target=target,
                pred=pred,
                block_info_full=block_info_full,
                delay_shape_eps=float(cfg.get("delay_shape_eps", 1e-8)),
                delay_shape_min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
                delay_shape_group_mode=str(cfg.get("delay_shape_group_mode", "celltype")),
            )
            target_rows_all.extend(target_rows)
            pred_rows_all.extend(pred_rows)
            top_rows_all.extend(top_rows)
            lr_rows_all.extend(lr_rows)

    target_df = pd.DataFrame(target_rows_all)
    pred_df = pd.DataFrame(pred_rows_all)
    top_df = pd.DataFrame(top_rows_all)
    lr_df = pd.DataFrame(lr_rows_all)

    target_summary_df = _summarize_groups(
        df=target_df,
        metrics=["target_delay_std", "target_delay_derivative_rms", "target_delay_centered_energy"],
        group_specs=[("all", None), ("sign_group", "sign_group"), ("type_name", "type_name"), ("cond_name", "cond_name"), ("unit_key", "unit_key")],
    )
    pred_summary_df = _summarize_groups(
        df=pred_df,
        metrics=[
            "delay_mse",
            "delay_shape_raw",
            "dc_error",
            "ac_error",
            "delay_std_ratio",
            "pred_delay_derivative_rms",
            "constant_delay_mse",
            "constant_delay_shape_raw",
            "constant_dc_error",
            "constant_ac_error",
            "constant_delay_std_ratio",
        ],
        group_specs=[("all", None), ("sign_group", "sign_group"), ("type_name", "type_name"), ("cond_name", "cond_name"), ("unit_key", "unit_key")],
        ratio_specs=[
            ("delay_std_ratio_agg", "pred_delay_std", "target_delay_std"),
            ("constant_delay_std_ratio_agg", "constant_delay_std", "target_delay_std"),
        ],
    )
    lr_summary_df = _summarize_groups(
        df=lr_df,
        metrics=[
            "delay_lr_sep_target",
            "delay_lr_sep_pred",
            "delay_lr_sep_ratio",
            "constant_delay_lr_sep_pred",
            "constant_delay_lr_sep_ratio",
        ],
        group_specs=[("all", None), ("sign_group", "sign_group"), ("type_name", "type_name"), ("unit_key", "unit_key")],
        ratio_specs=[
            ("delay_lr_sep_ratio_agg", "delay_lr_sep_pred", "delay_lr_sep_target"),
            ("constant_delay_lr_sep_ratio_agg", "constant_delay_lr_sep_pred", "delay_lr_sep_target"),
        ],
    )

    rel_group_rows: List[Dict[str, Any]] = []
    for name in sorted(group_den_sum):
        den = float(group_den_sum[name])
        current_num = float(group_num_sum.get(name, 0.0))
        const_num = float(const_group_num_sum.get(name, 0.0))
        rel_group_rows.append(
            {
                "group_name": str(name),
                "current_model_ac_rel_loss": _safe_ratio(current_num, den),
                "constant_delay_ac_rel_loss": _safe_ratio(const_num, den),
                "scale_sum": float(den),
            }
        )
    rel_group_df = pd.DataFrame(rel_group_rows)

    summary = {
        "label": str(label),
        "model_path": None if model_path is None else str(model_path),
        "eval_psth": _safe_ratio(loss_mask_sum, float(loss_mask_count)),
        "sample_mse": _safe_ratio(epoch_sq["sample"], float(epoch_count["sample"])),
        "delay_mse": _safe_ratio(epoch_sq["delay"], float(epoch_count["delay"])),
        "response_mse": _safe_ratio(epoch_sq["response"], float(epoch_count["response"])),
        "delay_shape_raw": _mean_or_nan(pred_df["delay_shape_raw"].tolist()),
        "dc_error": _safe_ratio(delay_dc_sum, float(delay_dc_count)),
        "ac_error": _mean_or_nan(pred_df["ac_error"].tolist()),
        "current_model_raw_shape_loss": _mean_or_nan(pred_df["delay_shape_raw"].tolist()),
        "constant_delay_raw_shape_loss": _mean_or_nan(pred_df["constant_delay_shape_raw"].tolist()),
        "current_model_ac_rel_loss": _safe_ratio(_mean_or_nan(pred_df["delay_shape_raw"].tolist()), _mean_or_nan(pred_df["target_delay_centered_energy"].tolist())),
        "constant_delay_ac_rel_loss": _safe_ratio(_mean_or_nan(pred_df["constant_delay_shape_raw"].tolist()), _mean_or_nan(pred_df["target_delay_centered_energy"].tolist())),
        "current_model_ac_rel_group_loss": _mean_or_nan(rel_group_df["current_model_ac_rel_loss"].tolist()) if not rel_group_df.empty else float("nan"),
        "constant_delay_ac_rel_group_loss": _mean_or_nan(rel_group_df["constant_delay_ac_rel_loss"].tolist()) if not rel_group_df.empty else float("nan"),
        "delay_std_ratio": float(pred_summary_df.loc[(pred_summary_df["group_family"] == "all") & (pred_summary_df["group_name"] == "all"), "delay_std_ratio_agg"].iloc[0]),
        "delay_lr_sep_target": float(lr_summary_df.loc[(lr_summary_df["group_family"] == "all") & (lr_summary_df["group_name"] == "all"), "delay_lr_sep_target"].iloc[0]),
        "delay_lr_sep_pred": float(lr_summary_df.loc[(lr_summary_df["group_family"] == "all") & (lr_summary_df["group_name"] == "all"), "delay_lr_sep_pred"].iloc[0]),
        "delay_lr_sep_ratio": float(lr_summary_df.loc[(lr_summary_df["group_family"] == "all") & (lr_summary_df["group_name"] == "all"), "delay_lr_sep_ratio_agg"].iloc[0]),
        "relative_ac_group_warnings": sorted(set(rel_group_warnings)),
        "relative_ac_group_constant_warnings": sorted(set(rel_const_group_warnings)),
    }

    out_paths: Dict[str, Any] = {}
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        out_paths["target_rows_csv"] = _save_df(target_df, os.path.join(out_dir, "target_delay_rows.csv"))
        out_paths["target_group_summary_csv"] = _save_df(target_summary_df, os.path.join(out_dir, "target_delay_group_summary.csv"))
        out_paths["prediction_rows_csv"] = _save_df(pred_df, os.path.join(out_dir, "prediction_vs_constant_rows.csv"))
        out_paths["prediction_group_summary_csv"] = _save_df(pred_summary_df, os.path.join(out_dir, "prediction_vs_constant_group_summary.csv"))
        out_paths["delay_lr_rows_csv"] = _save_df(lr_df, os.path.join(out_dir, "delay_lr_rows.csv"))
        out_paths["delay_lr_group_summary_csv"] = _save_df(lr_summary_df, os.path.join(out_dir, "delay_lr_group_summary.csv"))
        out_paths["relative_ac_group_csv"] = _save_df(rel_group_df, os.path.join(out_dir, "relative_ac_group_summary.csv"))
        health_json = save_blockrank_diagnostics(
            model=model,
            out_dir=out_dir,
            stem="current_model_health" if label == "current_model" else f"{label}_health",
            extra={
                "model_path": None if model_path is None else str(model_path),
                "delay_shape_group_mode": str(cfg.get("delay_shape_group_mode", "celltype")),
                **summary,
            },
        )
        out_paths["health_json"] = str(health_json)
    summary["artifacts"] = out_paths
    summary["target_summary_df"] = target_summary_df
    summary["pred_summary_df"] = pred_summary_df
    summary["lr_summary_df"] = lr_summary_df
    summary["top_df"] = top_df
    return summary


def _unit_test_relative_ac(out_dir: str) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    delay_mask = np.ones((5,), dtype=np.bool_)
    target = tch.tensor([[[0.40], [0.45], [0.50], [0.47], [0.43]]], dtype=tch.float32)
    pred_good = tch.tensor([[[0.41], [0.46], [0.51], [0.48], [0.44]]], dtype=tch.float32)
    pred_flat = tch.tensor([[[0.45], [0.45], [0.45], [0.45], [0.45]]], dtype=tch.float32)
    pred_shifted = tch.tensor([[[1.40], [1.45], [1.50], [1.47], [1.43]]], dtype=tch.float32)

    raw_good = _float_item(delay_centered_shape_loss(target, pred_good, delay_mask))
    raw_flat = _float_item(delay_centered_shape_loss(target, pred_flat, delay_mask))
    raw_shifted = _float_item(delay_centered_shape_loss(target, pred_shifted, delay_mask))
    rel_good, rel_good_details = compute_relative_ac_loss(target=target, pred=pred_good, delay_mask=delay_mask, scale_mode="global_ac_energy")
    rel_flat, rel_flat_details = compute_relative_ac_loss(target=target, pred=pred_flat, delay_mask=delay_mask, scale_mode="global_ac_energy")
    rel_shifted, rel_shifted_details = compute_relative_ac_loss(target=target, pred=pred_shifted, delay_mask=delay_mask, scale_mode="global_ac_energy")

    payload = {
        "target": [0.40, 0.45, 0.50, 0.47, 0.43],
        "pred_good": [0.41, 0.46, 0.51, 0.48, 0.44],
        "pred_flat": [0.45, 0.45, 0.45, 0.45, 0.45],
        "pred_shifted": [1.40, 1.45, 1.50, 1.47, 1.43],
        "raw_centered_loss": {
            "pred_good": float(raw_good),
            "pred_flat": float(raw_flat),
            "pred_shifted": float(raw_shifted),
        },
        "relative_ac_loss": {
            "pred_good": float(rel_good.detach().cpu().item()),
            "pred_flat": float(rel_flat.detach().cpu().item()),
            "pred_shifted": float(rel_shifted.detach().cpu().item()),
        },
        "details": {
            "pred_good": rel_good_details,
            "pred_flat": rel_flat_details,
            "pred_shifted": rel_shifted_details,
        },
        "checks": {
            "raw_good_lt_flat": bool(raw_good < raw_flat),
            "raw_shifted_lt_flat": bool(raw_shifted < raw_flat),
            "relative_flat_close_to_one": float(abs(float(rel_flat.detach().cpu().item()) - 1.0)),
            "relative_good_small": float(rel_good.detach().cpu().item()),
            "relative_shifted_small": float(rel_shifted.detach().cpu().item()),
        },
    }
    _write_json(payload, os.path.join(out_dir, "relative_ac_loss_unit_test.json"))
    return payload


def _compute_probe_loss_terms(
    *,
    model: CellTypeBlockRank1DaleCurrentRNN,
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    block_info_full: Dict[str, Any],
) -> Dict[str, Any]:
    loss_fn = LossAverageTrials()
    target = tch.nan_to_num(_select_psth_target(batch, str(cfg.get("psth_target_source", "trials_mean"))), nan=0.0, posinf=0.0, neginf=0.0)
    pred_full = model(batch["u"], h0=None, noise_std=0.0, return_rate=True)["rate"]
    pred = pred_full.index_select(dim=2, index=batch["idx_net"])
    epoch_masks = build_epoch_masks(
        batch["meta"],
        T=int(target.shape[1]),
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
    )
    raw_epoch_losses = _epoch_losses(loss_fn=loss_fn, target=target, pred=pred, epoch_masks=epoch_masks)
    norm_epoch_losses = _epoch_normalized_losses(loss_fn=loss_fn, target=target, pred=pred, epoch_masks=epoch_masks)
    main_raw = _weighted_mean_loss(raw_epoch_losses, _cfg_loss_epoch_weights(cfg))
    main_norm = _weighted_mean_loss(norm_epoch_losses, _cfg_loss_epoch_weights(cfg))
    group_args = _local_delay_group_args(
        batch=batch,
        block_info_full=block_info_full,
        group_mode=str(cfg.get("delay_shape_group_mode", "celltype")),
    )
    shape_raw = compute_delay_shape_loss_bundle(
        target=target,
        pred=pred,
        delay_mask=epoch_masks["delay"],
        loss_type=DELAY_SHAPE_LOSS_RAW,
        eps=float(cfg.get("delay_shape_eps", 1e-8)),
        min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
    )
    rel_global = compute_delay_shape_loss_bundle(
        target=target,
        pred=pred,
        delay_mask=epoch_masks["delay"],
        loss_type=DELAY_SHAPE_LOSS_REL_GLOBAL,
        eps=float(cfg.get("delay_shape_eps", 1e-8)),
        min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
    )
    rel_group = compute_delay_shape_loss_bundle(
        target=target,
        pred=pred,
        delay_mask=epoch_masks["delay"],
        loss_type=DELAY_SHAPE_LOSS_REL_GROUP,
        group_mode=str(group_args["group_mode"]),
        group_index=group_args["group_index"],
        group_names=group_args["group_names"],
        eps=float(cfg.get("delay_shape_eps", 1e-8)),
        min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
    )
    return {
        "target": target,
        "pred": pred,
        "epoch_masks": epoch_masks,
        "main_raw": main_raw,
        "main_norm": main_norm,
        "raw_epoch_losses": raw_epoch_losses,
        "norm_epoch_losses": norm_epoch_losses,
        "shape_raw": shape_raw,
        "rel_global": rel_global,
        "rel_group": rel_group,
    }


def _collect_grad_norms_for_loss(
    *,
    cfg: Dict[str, Any],
    state_dict: Dict[str, Any],
    batch: Dict[str, Any],
    block_info_full: Dict[str, Any],
    shared: Dict[str, Any],
    device: tch.device,
    loss_kind: str,
    lambda_value: float = 0.0,
) -> Dict[str, Any]:
    model = _build_model_from_cfg(
        cfg=cfg,
        n_total=int(shared["N_total"]),
        d_in=int(batch["u"].shape[-1]),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.zero_grad(set_to_none=True)
    terms = _compute_probe_loss_terms(model=model, batch=batch, cfg=cfg, block_info_full=block_info_full)
    if loss_kind == "main":
        loss = terms["main_raw"]
    elif loss_kind == "rel_global":
        loss = terms["rel_global"]["loss"]
    elif loss_kind == "rel_group":
        loss = terms["rel_group"]["loss"]
    elif loss_kind == "total_global":
        loss = terms["main_raw"] + float(lambda_value) * terms["rel_global"]["loss"]
    elif loss_kind == "total_group":
        loss = terms["main_raw"] + float(lambda_value) * terms["rel_group"]["loss"]
    else:
        raise ValueError(f"Unsupported loss_kind={loss_kind!r}")
    loss.backward()
    out = _grad_group_norms(model)
    out["loss_value"] = float(loss.detach().cpu().item())
    out["loss_kind"] = str(loss_kind)
    out["lambda"] = float(lambda_value)
    return out


def _recommend_lambdas(
    rows: Sequence[Dict[str, Any]],
    *,
    min_ratio: float = 0.05,
    max_ratio: float = 0.30,
    min_count: int = 2,
    max_count: int = 3,
) -> Dict[str, Any]:
    usable = [dict(r) for r in rows if math.isfinite(float(r["shape_grad_ratio"])) and float(r["shape_grad_ratio"]) > 0.0]
    in_band = [r for r in usable if float(r["shape_grad_ratio"]) >= float(min_ratio) and float(r["shape_grad_ratio"]) <= float(max_ratio)]
    if len(in_band) >= min_count:
        selected = sorted(in_band, key=lambda r: float(r["lambda"]))[:max_count]
    else:
        ranked = sorted(usable, key=lambda r: abs(float(r["shape_grad_ratio"]) - 0.12))
        selected = ranked[: max(min_count, min(len(ranked), max_count))]
    return {
        "target_ratio_range": [float(min_ratio), float(max_ratio)],
        "selected": [float(r["lambda"]) for r in selected],
        "rows": usable,
    }


def _loss_scale_and_gradient_check(
    *,
    cfg: Dict[str, Any],
    state_dict: Dict[str, Any],
    probe_batch: Dict[str, Any],
    block_info_full: Dict[str, Any],
    shared: Dict[str, Any],
    device: tch.device,
    out_dir: str,
    lambda_candidates: Sequence[float],
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    model = _build_model_from_cfg(
        cfg=cfg,
        n_total=int(shared["N_total"]),
        d_in=int(probe_batch["u"].shape[-1]),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    terms = _compute_probe_loss_terms(model=model, batch=probe_batch, cfg=cfg, block_info_full=block_info_full)
    centered_target = compute_delay_statistics(terms["target"], terms["epoch_masks"]["delay"])
    centered_pred = compute_delay_statistics(terms["pred"], terms["epoch_masks"]["delay"])

    scale_payload = {
        "probe_unit_key": str(probe_batch["unit_key"]),
        "L_main_raw_epoch_mse": float(terms["main_raw"].detach().cpu().item()),
        "L_epoch_normalized_mse": float(terms["main_norm"].detach().cpu().item()),
        "L_sample": float(terms["raw_epoch_losses"]["sample"].detach().cpu().item()),
        "L_delay": float(terms["raw_epoch_losses"]["delay"].detach().cpu().item()),
        "L_response": float(terms["raw_epoch_losses"]["response"].detach().cpu().item()),
        "L_delay_shape_raw": float(terms["shape_raw"]["loss"].detach().cpu().item()),
        "L_delay_ac_rel_global": float(terms["rel_global"]["loss"].detach().cpu().item()),
        "L_delay_ac_rel_group": float(terms["rel_group"]["loss"].detach().cpu().item()),
        "centered_target_mean_over_time_maxabs": float(centered_target["max_abs_centered_mean_over_time"].detach().cpu().item()),
        "centered_pred_mean_over_time_maxabs": float(centered_pred["max_abs_centered_mean_over_time"].detach().cpu().item()),
        "global_details": terms["rel_global"]["details"],
        "group_details": terms["rel_group"]["details"],
        "lambda_table": [],
    }
    main_val = float(terms["main_raw"].detach().cpu().item())
    for lam in lambda_candidates:
        lam_f = float(lam)
        scale_payload["lambda_table"].append(
            {
                "lambda": float(lam_f),
                "weighted_ac_loss_global": lam_f * float(terms["rel_global"]["loss"].detach().cpu().item()),
                "weighted_ac_loss_global_over_L_main": _safe_ratio(lam_f * float(terms["rel_global"]["loss"].detach().cpu().item()), main_val),
                "weighted_ac_loss_group": lam_f * float(terms["rel_group"]["loss"].detach().cpu().item()),
                "weighted_ac_loss_group_over_L_main": _safe_ratio(lam_f * float(terms["rel_group"]["loss"].detach().cpu().item()), main_val),
            }
        )
    scale_json = _write_json(scale_payload, os.path.join(out_dir, "loss_scale_check_relative_ac.json"))

    grad_main = _collect_grad_norms_for_loss(
        cfg=cfg,
        state_dict=state_dict,
        batch=probe_batch,
        block_info_full=block_info_full,
        shared=shared,
        device=device,
        loss_kind="main",
    )
    grad_global = _collect_grad_norms_for_loss(
        cfg=cfg,
        state_dict=state_dict,
        batch=probe_batch,
        block_info_full=block_info_full,
        shared=shared,
        device=device,
        loss_kind="rel_global",
    )
    grad_group = _collect_grad_norms_for_loss(
        cfg=cfg,
        state_dict=state_dict,
        batch=probe_batch,
        block_info_full=block_info_full,
        shared=shared,
        device=device,
        loss_kind="rel_group",
    )

    lambda_table_global: List[Dict[str, Any]] = []
    lambda_table_group: List[Dict[str, Any]] = []
    total_runs: List[Dict[str, Any]] = []
    main_norm = float(grad_main["total_grad_norm"])
    global_shape_norm = float(grad_global["total_grad_norm"])
    group_shape_norm = float(grad_group["total_grad_norm"])
    for lam in lambda_candidates:
        lam_f = float(lam)
        row_global = {
            "lambda": lam_f,
            "shape_grad_ratio": _safe_ratio(lam_f * global_shape_norm, main_norm),
            "weighted_shape_grad_norm": lam_f * global_shape_norm,
            "main_grad_norm": main_norm,
        }
        row_group = {
            "lambda": lam_f,
            "shape_grad_ratio": _safe_ratio(lam_f * group_shape_norm, main_norm),
            "weighted_shape_grad_norm": lam_f * group_shape_norm,
            "main_grad_norm": main_norm,
        }
        lambda_table_global.append(row_global)
        lambda_table_group.append(row_group)
        total_runs.append(
            {
                "loss_type": "relative_ac_global",
                "lambda": lam_f,
                "grad": _collect_grad_norms_for_loss(
                    cfg=cfg,
                    state_dict=state_dict,
                    batch=probe_batch,
                    block_info_full=block_info_full,
                    shared=shared,
                    device=device,
                    loss_kind="total_global",
                    lambda_value=lam_f,
                ),
            }
        )
        total_runs.append(
            {
                "loss_type": "relative_ac_group",
                "lambda": lam_f,
                "grad": _collect_grad_norms_for_loss(
                    cfg=cfg,
                    state_dict=state_dict,
                    batch=probe_batch,
                    block_info_full=block_info_full,
                    shared=shared,
                    device=device,
                    loss_kind="total_group",
                    lambda_value=lam_f,
                ),
            }
        )

    rec_global = _recommend_lambdas(lambda_table_global)
    rec_group = _recommend_lambdas(lambda_table_group)
    selected_loss_type = str(cfg.get("delay_shape_loss_type", DELAY_SHAPE_LOSS_REL_GLOBAL)).strip().lower()
    if selected_loss_type not in {DELAY_SHAPE_LOSS_REL_GLOBAL, DELAY_SHAPE_LOSS_REL_GROUP}:
        selected_loss_type = DELAY_SHAPE_LOSS_REL_GLOBAL
    selected_recs = rec_group if selected_loss_type == DELAY_SHAPE_LOSS_REL_GROUP else rec_global

    grad_payload = {
        "probe_unit_key": str(probe_batch["unit_key"]),
        "L_main_grad": grad_main,
        "L_delay_ac_rel_global_grad": grad_global,
        "L_delay_ac_rel_group_grad": grad_group,
        "lambda_table_global": lambda_table_global,
        "lambda_table_group": lambda_table_group,
        "combined_total_grad_runs": total_runs,
        "recommended_lambda_candidates": {
            "relative_ac_global": rec_global,
            "relative_ac_group": rec_group,
            "selected_loss_type": str(selected_loss_type),
            "selected_candidates": list(selected_recs["selected"]),
        },
    }
    grad_json = _write_json(grad_payload, os.path.join(out_dir, "gradient_check_relative_ac.json"))
    return {
        "loss_scale_json": str(scale_json),
        "gradient_json": str(grad_json),
        "loss_scale_payload": scale_payload,
        "gradient_payload": grad_payload,
        "recommended_lambda_candidates": list(selected_recs["selected"]),
        "selected_loss_type": str(selected_loss_type),
    }


def _plot_eval_mosaic(
    *,
    model: CellTypeBlockRank1DaleCurrentRNN,
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    out_png: str,
    title: str,
) -> str:
    with tch.no_grad():
        target = tch.nan_to_num(_select_psth_target(batch, str(cfg.get("psth_target_source", "trials_mean"))), nan=0.0, posinf=0.0, neginf=0.0)
        pred_full = model(batch["u"], h0=None, noise_std=0.0, return_rate=True)["rate"]
        pred = pred_full.index_select(dim=2, index=batch["idx_net"])
    mask_np, t_sec, ev_sec = compute_time_mask_tsec_zero_at_R(
        meta=batch["meta"],
        T=int(target.shape[1]),
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
    )
    mask_t = tch.as_tensor(mask_np.astype(np.bool_), device=target.device)
    mse_per_neuron = (pred[:, mask_t, :] - target[:, mask_t, :]).pow(2).mean(dim=(0, 1)).detach().cpu().numpy()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plot_psth_comparison_R0_with_events(
        t_sec=np.asarray(t_sec, dtype=float),
        psth_used=target[:, mask_t, :].detach().cpu(),
        rates_used=pred[:, mask_t, :].detach().cpu(),
        idx_neurons=list(range(int(target.shape[-1]))),
        mse_per_neuron=np.asarray(mse_per_neuron, dtype=float),
        cond_names=list(batch["meta"]["cond_names"]),
        out_path=out_png,
        ncols=min(6, max(1, int(target.shape[-1]))),
        title=str(title),
        event_sec=ev_sec,
        show_event_labels=True,
    )
    return str(out_png)


def _run_short_finetune(
    *,
    cfg: Dict[str, Any],
    units: List[Dict[str, Any]],
    block_info_full: Dict[str, Any],
    shared: Dict[str, Any],
    device: tch.device,
    checkpoint_path: str,
    loss_type: str,
    shape_lambda: float,
    epochs: int,
    seed: int,
    out_dir: str,
    probe_batch: Dict[str, Any],
    baseline_summary: Dict[str, Any],
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    _seed_all(seed, device)
    state_dict = _load_checkpoint_state(checkpoint_path, device)
    model = _build_model_from_cfg(
        cfg=cfg,
        n_total=int(shared["N_total"]),
        d_in=int(units[0]["u"].shape[-1]),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        device=device,
    )
    model.load_state_dict(state_dict, strict=True)
    model.train()

    opt = tch.optim.Adam(
        model.parameters(),
        lr=float(cfg.get("lr", 5e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    loss_fn = LossAverageTrials()
    weights = _cfg_loss_epoch_weights(cfg)
    eval_every = max(50, min(200, int(epochs)))
    history_rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for ep in range(int(epochs)):
        batch = units[ep % len(units)]
        opt.zero_grad(set_to_none=True)
        target = tch.nan_to_num(_select_psth_target(batch, str(cfg.get("psth_target_source", "trials_mean"))), nan=0.0, posinf=0.0, neginf=0.0)
        pred_full = model(batch["u"], h0=None, noise_std=0.0, return_rate=True)["rate"]
        pred = pred_full.index_select(dim=2, index=batch["idx_net"])
        epoch_masks = build_epoch_masks(
            batch["meta"],
            T=int(target.shape[1]),
            sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
            resp_sec=float(cfg.get("resp_sec", 2.0)),
        )
        raw_epoch_losses = _epoch_losses(loss_fn=loss_fn, target=target, pred=pred, epoch_masks=epoch_masks)
        norm_epoch_losses = _epoch_normalized_losses(loss_fn=loss_fn, target=target, pred=pred, epoch_masks=epoch_masks)
        main_loss = _select_main_loss(cfg=cfg, raw_epoch_losses=raw_epoch_losses, norm_epoch_losses=norm_epoch_losses)
        group_args = _local_delay_group_args(
            batch=batch,
            block_info_full=block_info_full,
            group_mode=str(cfg.get("delay_shape_group_mode", "celltype")),
        )
        shape_bundle = compute_delay_shape_loss_bundle(
            target=target,
            pred=pred,
            delay_mask=epoch_masks["delay"],
            loss_type=str(loss_type),
            group_mode=str(group_args["group_mode"]),
            group_index=group_args["group_index"],
            group_names=group_args["group_names"],
            eps=float(cfg.get("delay_shape_eps", 1e-8)),
            min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
        )
        reg_block = model.recurrent_regularization_loss(
            A_l2=float(cfg.get("A_l2", 0.0)),
            uv_l2=float(cfg.get("uv_l2", 0.0)),
        )
        total_loss = main_loss + reg_block + float(shape_lambda) * shape_bundle["loss"]
        total_loss.backward()
        if cfg.get("grad_clip", None) not in (None, 0):
            tch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["grad_clip"]))
        opt.step()

        row = {
            "epoch": int(ep + 1),
            "train_total": float(total_loss.detach().cpu().item()),
            "train_main": float(main_loss.detach().cpu().item()),
            "train_delay_shape": float(shape_bundle["loss"].detach().cpu().item()),
            "train_weighted_delay_shape": float(shape_lambda) * float(shape_bundle["loss"].detach().cpu().item()),
            "train_reg_block": float(reg_block.detach().cpu().item()),
            "train_sample": float(raw_epoch_losses["sample"].detach().cpu().item()),
            "train_delay": float(raw_epoch_losses["delay"].detach().cpu().item()),
            "train_response": float(raw_epoch_losses["response"].detach().cpu().item()),
        }
        if ep == 0 or ((ep + 1) % eval_every == 0) or (ep + 1 == int(epochs)):
            eval_now = _evaluate_model_on_units(
                model=model,
                units=units,
                cfg=cfg,
                block_info_full=block_info_full,
                out_dir=None,
                model_path=None,
                label=f"finetune_ep{ep+1:06d}",
            )
            row.update(
                {
                    "eval_psth": float(eval_now["eval_psth"]),
                    "delay_mse": float(eval_now["delay_mse"]),
                    "delay_shape_raw": float(eval_now["delay_shape_raw"]),
                    "delay_ac_rel_loss": float(eval_now["current_model_ac_rel_loss"]),
                    "delay_std_ratio": float(eval_now["delay_std_ratio"]),
                    "delay_lr_sep_ratio": float(eval_now["delay_lr_sep_ratio"]),
                    "response_mse": float(eval_now["response_mse"]),
                }
            )
        history_rows.append(row)
        if ep == 0 or ((ep + 1) % max(1, int(epochs) // 5) == 0) or (ep + 1 == int(epochs)):
            print(
                f"[finetune:{loss_type}:lam={shape_lambda}] ep={ep+1}/{epochs} "
                f"total={row['train_total']:.6f} main={row['train_main']:.6f} "
                f"shape={row['train_delay_shape']:.6f}",
                flush=True,
            )

    history_df = pd.DataFrame(history_rows)
    history_csv = _save_df(history_df, os.path.join(out_dir, "training_history.csv"))

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(history_df["epoch"], history_df["train_total"], label="train_total", color="tab:blue")
    ax1.plot(history_df["epoch"], history_df["train_delay_shape"], label="train_delay_shape", color="tab:orange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    if "delay_std_ratio" in history_df.columns and history_df["delay_std_ratio"].notna().any():
        ax2.plot(history_df["epoch"], history_df["delay_std_ratio"], label="delay_std_ratio", color="tab:green")
    ax2.set_ylabel("Delay Std Ratio")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best", frameon=False)
    fig.tight_layout()
    curve_png = os.path.join(out_dir, "training_curves.png")
    fig.savefig(curve_png, dpi=160)
    plt.close(fig)

    final_eval_dir = os.path.join(out_dir, "eval_final")
    final_eval = _evaluate_model_on_units(
        model=model,
        units=units,
        cfg=cfg,
        block_info_full=block_info_full,
        out_dir=final_eval_dir,
        model_path=checkpoint_path,
        label="finetune_final",
    )
    final_ckpt = os.path.join(out_dir, "finetuned.pt")
    tch.save({"model": model.state_dict()}, final_ckpt)
    mosaic_png = _plot_eval_mosaic(
        model=model,
        batch=probe_batch,
        cfg=cfg,
        out_png=os.path.join(out_dir, "psth_mosaic.png"),
        title=f"fine-tune {loss_type} lambda={shape_lambda} epochs={epochs}",
    )

    summary = {
        "checkpoint_path": str(checkpoint_path),
        "loss_type": str(loss_type),
        "shape_lambda": float(shape_lambda),
        "epochs": int(epochs),
        "elapsed_sec": float(time.time() - t0),
        "history_csv": str(history_csv),
        "curve_png": str(curve_png),
        "final_checkpoint": str(final_ckpt),
        "mosaic_png": str(mosaic_png),
        "response_mse_delta_vs_baseline": float(final_eval["response_mse"]) - float(baseline_summary["response_mse"]),
        **{k: v for k, v in final_eval.items() if not isinstance(v, pd.DataFrame)},
    }
    summary_json = _write_json(summary, os.path.join(out_dir, "summary.json"))
    summary["summary_json"] = str(summary_json)
    return summary


def _decide_long_training(
    *,
    baseline_summary: Dict[str, Any],
    fine_tune_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    reasons: List[str] = []
    passing: List[Dict[str, Any]] = []
    base_eval = float(baseline_summary["eval_psth"])
    base_delay_std = float(baseline_summary["delay_std_ratio"])
    base_delay_ac = float(baseline_summary["current_model_ac_rel_loss"])
    base_resp = float(baseline_summary["response_mse"])
    for row in fine_tune_rows:
        dale_zero = int(row.get("dale_violation_count", 1)) == 0
        ac_improved = float(row["current_model_ac_rel_loss"]) < (base_delay_ac - 0.02)
        std_improved = float(row["delay_std_ratio"]) > max(base_delay_std + 0.01, base_delay_std * 2.0)
        eval_ok = float(row["eval_psth"]) <= base_eval * 1.10
        lr_ok = abs(float(row["delay_lr_sep_ratio"]) - 1.0) <= 0.20
        resp_ok = float(row["response_mse"]) <= max(base_resp * 1.25, base_resp + 0.002)
        if dale_zero and ac_improved and std_improved and eval_ok and lr_ok and resp_ok:
            passing.append(row)
    if len(passing) == 0:
        reasons.append("No short fine-tune simultaneously reduced relative AC loss, increased delay_std_ratio, preserved delay_lr_sep_ratio, and kept eval_psth/response MSE stable.")
        return {"recommend_long_training": False, "reasons": reasons, "passing_runs": []}
    reasons.append(f"{len(passing)} short fine-tune run(s) met the delay-dynamics gate.")
    return {
        "recommend_long_training": True,
        "reasons": reasons,
        "passing_runs": [{"loss_type": r["loss_type"], "shape_lambda": r["shape_lambda"], "epochs": r["epochs"], "summary_json": r["summary_json"]} for r in passing],
    }


def _run_preflight(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = _load_json(str(args.config))
    requested_device = str(args.device or cfg.get("device", "auto"))
    dev, runtime_info = _configure_runtime_device(
        device=requested_device,
        require_cuda=bool(args.require_cuda),
        enable_tf32=bool(cfg.get("enable_tf32", True)),
        cudnn_benchmark=bool(cfg.get("cudnn_benchmark", True)),
        matmul_precision=str(cfg.get("matmul_precision", "high")),
    )
    out_dir = str(args.out_dir or _default_out_dir())
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[INFO] Preflight device requested={requested_device} resolved={runtime_info['resolved_device']} "
        f"cuda_available={runtime_info['cuda_available']}",
        flush=True,
    )

    units, shared, block_info_obs, block_info_full, _ = _load_units_and_block_info(cfg, dev)
    for batch in units:
        batch["meta"]["_sample_ignore_ms"] = float(cfg.get("sample_ignore_ms", 50.0))
        batch["meta"]["_resp_sec"] = float(cfg.get("resp_sec", 2.0))
        batch["meta"]["_psth_target_source"] = str(cfg.get("psth_target_source", "trials_mean"))

    units_by_key = {str(b["unit_key"]): b for b in units}
    lambda_candidates = _parse_float_csv(args.lambda_candidates)
    finetune_epochs = _parse_int_csv(args.finetune_epochs)
    checkpoint_path = str(args.model)
    state_dict = _load_checkpoint_state(checkpoint_path, dev)

    target_rows_all: List[Dict[str, Any]] = []
    top_rows_all: List[Dict[str, Any]] = []
    for batch in units:
        target = tch.nan_to_num(_select_psth_target(batch, str(cfg.get("psth_target_source", "trials_mean"))), nan=0.0, posinf=0.0, neginf=0.0)
        target_rows, _, top_rows, _ = _build_target_prediction_tables(
            batch=batch,
            target=target,
            pred=target.clone(),
            block_info_full=block_info_full,
            delay_shape_eps=float(cfg.get("delay_shape_eps", 1e-8)),
            delay_shape_min_scale=float(cfg.get("delay_shape_min_scale", 1e-6)),
            delay_shape_group_mode=str(cfg.get("delay_shape_group_mode", "celltype")),
        )
        target_rows_all.extend(target_rows)
        top_rows_all.extend(top_rows)

    target_df = pd.DataFrame(target_rows_all)
    target_summary_df = _summarize_groups(
        df=target_df,
        metrics=["target_delay_std", "target_delay_derivative_rms", "target_delay_centered_energy"],
        group_specs=[("all", None), ("sign_group", "sign_group"), ("type_name", "type_name"), ("cond_name", "cond_name"), ("unit_key", "unit_key")],
    )
    target_dir = os.path.join(out_dir, "target_delay_diagnostics")
    target_rows_csv = _save_df(target_df, os.path.join(target_dir, "target_delay_rows.csv"))
    target_summary_csv = _save_df(target_summary_df, os.path.join(target_dir, "target_delay_group_summary.csv"))
    target_top_csv = _save_df(pd.DataFrame(top_rows_all), os.path.join(target_dir, "target_delay_top_rows.csv"))
    target_plot_png = _plot_top_target_neurons(
        records=list(top_rows_all),
        units_by_key=units_by_key,
        out_png=os.path.join(target_dir, "target_delay_top_neurons.png"),
        top_n=int(args.plot_top_n),
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
    )

    current_eval_dir = os.path.join(out_dir, "current_model_eval")
    current_model = _build_model_from_cfg(
        cfg=cfg,
        n_total=int(shared["N_total"]),
        d_in=int(units[0]["u"].shape[-1]),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        device=dev,
    )
    current_model.load_state_dict(state_dict, strict=True)
    current_summary = _evaluate_model_on_units(
        model=current_model,
        units=units,
        cfg=cfg,
        block_info_full=block_info_full,
        out_dir=current_eval_dir,
        model_path=checkpoint_path,
        label="current_model",
    )

    unit_test_payload = _unit_test_relative_ac(out_dir)
    probe_batch = _pick_probe_batch(
        units,
        block_info_full,
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
        psth_target_source=str(cfg.get("psth_target_source", "trials_mean")),
    )
    loss_grad = _loss_scale_and_gradient_check(
        cfg=cfg,
        state_dict=state_dict,
        probe_batch=probe_batch,
        block_info_full=block_info_full,
        shared=shared,
        device=dev,
        out_dir=os.path.join(out_dir, "loss_scale_and_grad_check"),
        lambda_candidates=lambda_candidates,
    )

    selected_loss_type = str(args.finetune_loss_type or loss_grad["selected_loss_type"] or DELAY_SHAPE_LOSS_REL_GLOBAL)
    if selected_loss_type not in {DELAY_SHAPE_LOSS_REL_GLOBAL, DELAY_SHAPE_LOSS_REL_GROUP}:
        selected_loss_type = DELAY_SHAPE_LOSS_REL_GLOBAL
    recommended = list(loss_grad["recommended_lambda_candidates"])
    if len(recommended) == 0:
        recommended = list(lambda_candidates[:2])
    max_lambdas = max(2, int(args.max_finetune_lambdas))
    selected_lambdas = recommended[: max_lambdas]

    short_dir = os.path.join(out_dir, "short_finetune")
    short_rows: List[Dict[str, Any]] = []
    for lam in selected_lambdas:
        for epochs in finetune_epochs:
            run_dir = os.path.join(short_dir, f"{selected_loss_type}_lambda{str(lam).replace('.', 'p')}_ep{epochs}")
            summary = _run_short_finetune(
                cfg={**cfg, "delay_shape_group_mode": str(args.finetune_group_mode or cfg.get("delay_shape_group_mode", "celltype"))},
                units=units,
                block_info_full=block_info_full,
                shared=shared,
                device=dev,
                checkpoint_path=checkpoint_path,
                loss_type=selected_loss_type,
                shape_lambda=float(lam),
                epochs=int(epochs),
                seed=int(args.seed),
                out_dir=run_dir,
                probe_batch=probe_batch,
                baseline_summary=current_summary,
            )
            short_rows.append(summary)
    short_df = pd.DataFrame(short_rows)
    short_csv = _save_df(short_df, os.path.join(short_dir, "short_finetune_summary.csv"))

    recommendation = _decide_long_training(
        baseline_summary=current_summary,
        fine_tune_rows=short_rows,
    )

    summary_payload = {
        "config_path": str(args.config),
        "checkpoint_path": str(checkpoint_path),
        "out_dir": str(out_dir),
        "runtime": runtime_info,
        "loss_type": str(selected_loss_type),
        "delay_shape_group_mode": str(args.finetune_group_mode or cfg.get("delay_shape_group_mode", "celltype")),
        "lambda_candidates_input": list(lambda_candidates),
        "recommended_lambda_candidates": list(selected_lambdas),
        "observed_type_counts": dict(block_info_obs["observed_type_counts"]),
        "full_type_counts": dict(block_info_full["full_type_counts"]),
        "target_delay_diagnostics": {
            "rows_csv": str(target_rows_csv),
            "group_summary_csv": str(target_summary_csv),
            "top_rows_csv": str(target_top_csv),
            "top_neuron_plot_png": None if target_plot_png is None else str(target_plot_png),
        },
        "relative_ac_loss_unit_test": unit_test_payload,
        "current_model_eval": {
            k: v
            for k, v in current_summary.items()
            if k not in {"target_summary_df", "pred_summary_df", "lr_summary_df", "top_df"}
        },
        "constant_delay_baseline_relative_ac": {
            "constant_delay_ac_rel_loss": float(current_summary["constant_delay_ac_rel_loss"]),
            "constant_delay_ac_rel_group_loss": float(current_summary["constant_delay_ac_rel_group_loss"]),
            "current_model_ac_rel_loss": float(current_summary["current_model_ac_rel_loss"]),
            "current_model_ac_rel_group_loss": float(current_summary["current_model_ac_rel_group_loss"]),
            "current_model_raw_shape_loss": float(current_summary["current_model_raw_shape_loss"]),
            "constant_delay_raw_shape_loss": float(current_summary["constant_delay_raw_shape_loss"]),
        },
        "loss_scale_table": {
            "json": str(loss_grad["loss_scale_json"]),
            "rows": loss_grad["loss_scale_payload"]["lambda_table"],
        },
        "gradient_ratio_table": {
            "json": str(loss_grad["gradient_json"]),
            "selected_loss_type": str(loss_grad["selected_loss_type"]),
            "global": loss_grad["gradient_payload"]["lambda_table_global"],
            "group": loss_grad["gradient_payload"]["lambda_table_group"],
            "recommended_lambda_candidates": loss_grad["gradient_payload"]["recommended_lambda_candidates"],
        },
        "short_fine_tune_result_table": {
            "csv": str(short_csv),
            "rows": short_rows,
        },
        "recommend_long_training": bool(recommendation["recommend_long_training"]),
        "recommendation_reasons": list(recommendation["reasons"]),
        "passing_runs": list(recommendation["passing_runs"]),
    }
    summary_json = _write_json(summary_payload, os.path.join(out_dir, "preflight_summary_relative_ac.json"))
    summary_payload["summary_json"] = str(summary_json)
    print(f"[OK] Saved relative-AC preflight summary -> {summary_json}", flush=True)
    return summary_payload


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to block-rank1 JSON config.")
    ap.add_argument("--model", required=True, help="Checkpoint path for current best model.")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--require_cuda", action="store_true")
    ap.add_argument("--plot_top_n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--lambda_candidates", default="0.03,0.1,0.3,1.0,3.0")
    ap.add_argument("--finetune_epochs", default="500,2000")
    ap.add_argument("--max_finetune_lambdas", type=int, default=3)
    ap.add_argument("--finetune_loss_type", default=None)
    ap.add_argument("--finetune_group_mode", default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    _run_preflight(args)


if __name__ == "__main__":
    main()
