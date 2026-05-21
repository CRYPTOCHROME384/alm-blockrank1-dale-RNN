from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as tch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from eval_current_alm import compute_time_mask_tsec_zero_at_R, plot_psth_comparison_R0_with_events
from losses import LossAverageTrials
from models import CellTypeBlockRank1DaleCurrentRNN
from training_current import _kernel_size_from_bin_ms, _preload_units_from_registry, _time_bin_smooth_ctn
from training_blockrank1_dale import (
    DEFAULT_DIAG_SUBDIR_EVAL,
    DEFAULT_PSTH_SMOOTHING,
    DEFAULT_VIZ_SMOOTH_BIN_MS,
    EPOCH_NAMES,
    _normalize_loss_epoch_weights,
    _select_psth_target,
)
from blockrank_dale_utils import (
    build_by_unit_from_registry_dataframe,
    build_epoch_masks,
    build_full_block_type_info,
    compute_delay_ratio_metrics,
    infer_block_type_info_from_registry_dataframe,
    load_registry_dataframe,
    local_sign_masks_from_idx_net,
    local_type_masks_from_idx_net,
    save_blockrank_diagnostics,
    subset_and_reindex_registry_dataframe,
)


def _load_json_if_exists(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None or not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _runtime_cfg_view(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(cfg, dict):
        return {}
    rr = cfg.get("_resolved_runtime", None)
    if isinstance(rr, dict):
        return rr
    return dict(cfg)


def _merge_missing(dst: Dict[str, Any], src: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(src, dict):
        return dst
    for k, v in src.items():
        dst.setdefault(k, v)
    return dst


def _strip_model_suffix(model_path: str) -> str:
    stem = os.path.splitext(model_path)[0]
    for suffix in (".best", ".latest"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _load_saved_run_config(model_path: str, params_path: Optional[str]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    base = _strip_model_suffix(model_path)
    model_dir = os.path.dirname(model_path)

    for cand in [
        base + ".train_config.json",
        base + ".meta.json",
        os.path.join(model_dir, "parameters.launch.json"),
        params_path,
    ]:
        obj = _load_json_if_exists(cand)
        if obj is not None:
            _merge_missing(cfg, _runtime_cfg_view(obj))
            _merge_missing(cfg, obj if cand and cand.endswith(".meta.json") else None)
    return cfg


def _infer_device(run_cfg: Dict[str, Any]) -> tch.device:
    device_str = str(run_cfg.get("device", "cuda" if tch.cuda.is_available() else "cpu"))
    return tch.device(device_str if (device_str == "cpu" or tch.cuda.is_available()) else "cpu")


def _build_model_and_units_for_eval(
    *,
    model_path: str,
    params_path: Optional[str],
    registry_dir_override: Optional[str],
    animal_override: Optional[str],
) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any], Dict[str, Any], Dict[str, Any], tch.device]:
    run_cfg = _load_saved_run_config(model_path=model_path, params_path=params_path)
    registry_dir = str(registry_dir_override or run_cfg.get("registry_dir", ""))
    animal = str(animal_override or run_cfg.get("animal", ""))
    if registry_dir == "" or animal == "":
        raise ValueError("Eval requires registry_dir and animal, either from CLI overrides or saved run config.")

    device = _infer_device(run_cfg)
    registry_df_full, registry_csv = load_registry_dataframe(registry_dir=registry_dir, animal=animal)
    registry_df, selected_unit_keys, reindex_map = subset_and_reindex_registry_dataframe(
        registry_df_full,
        max_sessions=(None if run_cfg.get("max_sessions", None) in (None, "", 0) else int(run_cfg.get("max_sessions"))),
    )
    by_unit = build_by_unit_from_registry_dataframe(registry_df)

    units, shared = _preload_units_from_registry(
        by_unit=by_unit,
        n_exc_virtual=int(run_cfg.get("n_exc_virtual", 0)),
        device=device,
        cond_filter=run_cfg.get("cond_filter", None),
        max_time=run_cfg.get("max_time", None),
        psth_bin_ms=float(run_cfg.get("psth_bin_ms", 200.0)),
        sample_ignore_ms=float(run_cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(run_cfg.get("resp_sec", 2.0)),
        use_trials=bool(run_cfg.get("use_trials", True)),
        strict_trials_align=bool(run_cfg.get("strict_trials_align", True)),
        trials_root=run_cfg.get("trials_root", None),
        trials_path_mode=str(run_cfg.get("trials_path_mode", "auto_from_stage1")),
        trial_keys=run_cfg.get("trial_keys", None),
        debug_trials_align=bool(run_cfg.get("debug_trials_align", False)),
        phase3_precompute_var_real=False,
        var_unbiased=False,
        min_trials_for_var_real=3,
    )

    block_info_obs = infer_block_type_info_from_registry_dataframe(
        registry_df,
        celltype_mode=str(run_cfg.get("celltype_mode", "broadE_inh_subclass")),
        registry_label_cols=run_cfg.get("block_registry_label_cols", None),
        inh_min_count=int(run_cfg.get("block_celltype_min_count", 10)),
        other_inh_label=str(run_cfg.get("other_inh_label", "I_other")),
    )
    block_info_full = build_full_block_type_info(block_info_obs, n_exc_virtual=int(run_cfg.get("n_exc_virtual", 0)))

    net = CellTypeBlockRank1DaleCurrentRNN(
        N=int(shared["N_total"]),
        D_in=int(units[0]["u"].shape[-1]),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        dt=float(run_cfg.get("dt", 0.03436)),
        tau=float(run_cfg.get("tau", 0.01)),
        substeps=int(run_cfg.get("substeps", 6)),
        nonlinearity=str(run_cfg.get("nonlinearity", "tanh")),
        block_rank=int(run_cfg.get("block_rank", 1)),
        factor_nonlinearity=str(run_cfg.get("factor_nonlinearity", "softplus")),
        A_nonlinearity=str(run_cfg.get("A_nonlinearity", "softplus")),
        normalize_uv=bool(run_cfg.get("normalize_uv", True)),
        eps=float(run_cfg.get("eps", 1e-8)),
        init_A=float(run_cfg.get("init_A", 0.10)),
        init_factor_scale=float(run_cfg.get("init_factor_scale", 0.02)),
        device=device,
    ).to(device)

    ckpt = tch.load(model_path, map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and isinstance(ckpt.get("model", None), dict) else ckpt
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        raise RuntimeError(f"Missing model keys during eval load: {missing}")
    if len(unexpected) > 0:
        raise RuntimeError(f"Unexpected model keys during eval load: {unexpected}")
    net.eval()

    run_cfg["_resolved_eval_registry_csv"] = str(registry_csv)
    run_cfg["_resolved_eval_selected_unit_keys"] = list(selected_unit_keys)
    run_cfg["_resolved_eval_reindex_size"] = int(len(reindex_map))
    return net, units, shared, block_info_full, run_cfg, device


def _weighted_mean_from_sum_count(sum_value: float, count_value: int) -> float:
    return float(sum_value) / float(count_value) if int(count_value) > 0 else float("nan")


def _resolve_aux_dir(base_dir: str, subdir: Optional[str]) -> str:
    if subdir in (None, ""):
        return str(base_dir)
    subdir_str = str(subdir)
    return subdir_str if os.path.isabs(subdir_str) else os.path.join(str(base_dir), subdir_str)


def _align_picked_records_for_mosaic(picked: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
    if len(picked) == 0:
        return picked, None

    common_keys = set(np.round(np.asarray(picked[0]["t_sec"], dtype=float), 8).tolist())
    for rec in picked[1:]:
        common_keys &= set(np.round(np.asarray(rec["t_sec"], dtype=float), 8).tolist())
    if len(common_keys) == 0:
        return [], None

    common_t = np.asarray(sorted(common_keys), dtype=float)
    aligned: List[Dict[str, Any]] = []
    for rec in picked:
        t_arr = np.asarray(rec["t_sec"], dtype=float)
        key_to_idx = {float(np.round(t, 8)): i for i, t in enumerate(t_arr.tolist())}
        keep_idx = [key_to_idx[float(np.round(t, 8))] for t in common_t.tolist()]
        keep_idx_t = tch.as_tensor(keep_idx, dtype=tch.long)
        rec_aligned = dict(rec)
        rec_aligned["t_sec"] = common_t
        rec_aligned["target"] = rec["target"].index_select(dim=1, index=keep_idx_t)
        rec_aligned["pred"] = rec["pred"].index_select(dim=1, index=keep_idx_t)
        if "target_viz" in rec:
            rec_aligned["target_viz"] = rec["target_viz"].index_select(dim=1, index=keep_idx_t)
        if "pred_viz" in rec:
            rec_aligned["pred_viz"] = rec["pred_viz"].index_select(dim=1, index=keep_idx_t)
        aligned.append(rec_aligned)
    return aligned, common_t


def eval_all_units_blockrank1_dale(
    *,
    model_path: str,
    params_path: Optional[str],
    registry_dir_override: Optional[str],
    animal_override: Optional[str],
    out_dir: str,
    plot_n: int = 72,
    plot_cols: int = 6,
) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)

    net, units, shared, block_info_full, run_cfg, device = _build_model_and_units_for_eval(
        model_path=model_path,
        params_path=params_path,
        registry_dir_override=registry_dir_override,
        animal_override=animal_override,
    )
    psth_target_source = str(run_cfg.get("psth_target_source", "trials_mean"))
    diag_out_dir = _resolve_aux_dir(out_dir, run_cfg.get("diag_subdir_eval", DEFAULT_DIAG_SUBDIR_EVAL))
    viz_smooth_bin_ms = max(float(run_cfg.get("viz_smooth_bin_ms", DEFAULT_VIZ_SMOOTH_BIN_MS)), float(run_cfg.get("psth_bin_ms", 200.0)))
    psth_kernel_frames = int(_kernel_size_from_bin_ms(float(shared["fps"]), float(run_cfg.get("psth_bin_ms", 200.0))))
    viz_kernel_frames = int(_kernel_size_from_bin_ms(float(shared["fps"]), float(viz_smooth_bin_ms)))
    loss_epoch_weights = _normalize_loss_epoch_weights(run_cfg.get("loss_epoch_weights", None))
    loss_fn = LossAverageTrials()
    model_tag = os.path.splitext(os.path.basename(model_path))[0]

    per_unit_rows: List[Dict[str, Any]] = []
    top_heap: List[Tuple[float, int, Dict[str, Any]]] = []
    heap_counter = 0

    sign_acc = {
        "excitatory": {"sum": 0.0, "count": 0},
        "inhibitory": {"sum": 0.0, "count": 0},
    }
    type_acc = {str(name): {"sum": 0.0, "count": 0} for name in block_info_full["type_names"]}
    epoch_acc = {name: {"sum": 0.0, "count": 0} for name in ["sample", "delay", "response"]}
    delay_acc = {
        "all": {"n_cells": 0},
        "excitatory": {"n_cells": 0},
        "inhibitory": {"n_cells": 0},
    }
    delay_fields = [
        "delay_std_target",
        "delay_std_pred",
        "delay_mean_abs_target",
        "delay_mean_abs_pred",
        "sample_std_target",
        "sample_std_pred",
        "resp_std_target",
        "resp_std_pred",
        "delay_lr_sep_target",
        "delay_lr_sep_pred",
    ]
    for g in delay_acc.values():
        for field in delay_fields:
            g[field] = 0.0

    print(f"[INFO] device={device}", flush=True)
    print(f"[INFO] model_type={run_cfg.get('model_type', 'unknown')}", flush=True)
    print(f"[INFO] psth_target_source={psth_target_source}", flush=True)
    print(
        f"[INFO] psth_smoothing={run_cfg.get('psth_smoothing', DEFAULT_PSTH_SMOOTHING)} "
        f"psth_bin_ms={float(run_cfg.get('psth_bin_ms', 200.0)):.1f} "
        f"psth_kernel_frames={int(psth_kernel_frames)} "
        f"viz_smooth_bin_ms={float(viz_smooth_bin_ms):.1f} "
        f"viz_kernel_frames={int(viz_kernel_frames)} "
        f"loss_epoch_weights={loss_epoch_weights}",
        flush=True,
    )
    print(f"[INFO] full_cell_type_counts={block_info_full['full_type_counts']}", flush=True)
    print(f"[INFO] block_list={block_info_full['block_list']}", flush=True)

    with tch.no_grad():
        for batch in units:
            u = tch.nan_to_num(batch["u"], nan=0.0, posinf=0.0, neginf=0.0)
            idx_net = batch["idx_net"]
            target = tch.nan_to_num(_select_psth_target(batch, psth_target_source), nan=0.0, posinf=0.0, neginf=0.0)
            pred_full = net(u, h0=None, noise_std=0.0, return_rate=True)["rate"]
            pred = pred_full.index_select(dim=2, index=idx_net)

            loss_mask = batch["time_mask"]
            diff_loss = (pred[:, loss_mask, :] - target[:, loss_mask, :]).pow(2)
            unit_mse = float(diff_loss.mean().detach().cpu().item())
            mse_per_neuron = diff_loss.mean(dim=(0, 1)).detach().cpu().numpy()

            per_unit_rows.append(
                {
                    "unit_key": str(batch["unit_key"]),
                    "npz_path": str(batch["npz_path"]),
                    "K": int(idx_net.numel()),
                    "loss_mse": float(unit_mse),
                    "mse_neuron_mean": float(np.mean(mse_per_neuron)),
                    "mse_neuron_median": float(np.median(mse_per_neuron)),
                }
            )

            sign_masks = local_sign_masks_from_idx_net(
                idx_net,
                full_is_exc=np.asarray(block_info_full["full_is_exc"], dtype=np.bool_),
            )
            type_masks = local_type_masks_from_idx_net(
                idx_net,
                full_type_index=np.asarray(block_info_full["full_type_index"], dtype=np.int64),
                type_names=list(block_info_full["type_names"]),
            )

            for sign_name, sign_mask in sign_masks.items():
                if int(sign_mask.sum().item()) <= 0:
                    continue
                diff_group = diff_loss[:, :, sign_mask]
                sign_acc[sign_name]["sum"] += float(diff_group.sum().detach().cpu().item())
                sign_acc[sign_name]["count"] += int(diff_group.numel())

            for type_name, type_mask in type_masks.items():
                if int(type_mask.sum().item()) <= 0:
                    continue
                diff_group = diff_loss[:, :, type_mask]
                type_acc[type_name]["sum"] += float(diff_group.sum().detach().cpu().item())
                type_acc[type_name]["count"] += int(diff_group.numel())

            epoch_masks = build_epoch_masks(
                batch["meta"],
                T=int(target.shape[1]),
                sample_ignore_ms=float(run_cfg.get("sample_ignore_ms", 50.0)),
                resp_sec=float(run_cfg.get("resp_sec", 2.0)),
            )
            for epoch_name in ["sample", "delay", "response"]:
                epoch_mask_np = np.asarray(epoch_masks[epoch_name], dtype=np.bool_)
                if int(epoch_mask_np.sum()) <= 0:
                    continue
                epoch_mask_t = tch.as_tensor(epoch_mask_np, device=target.device)
                diff_epoch = (pred[:, epoch_mask_t, :] - target[:, epoch_mask_t, :]).pow(2)
                epoch_acc[epoch_name]["sum"] += float(diff_epoch.sum().detach().cpu().item())
                epoch_acc[epoch_name]["count"] += int(diff_epoch.numel())

            delay_metrics = compute_delay_ratio_metrics(
                target=target,
                pred=pred,
                epoch_masks=epoch_masks,
                local_masks={
                    "all": tch.ones(int(idx_net.numel()), dtype=tch.bool, device=idx_net.device),
                    "excitatory": sign_masks["excitatory"],
                    "inhibitory": sign_masks["inhibitory"],
                },
            )
            for group_name, group_metrics in delay_metrics.items():
                n_cells = int(group_metrics["n_cells"])
                if n_cells <= 0:
                    continue
                delay_acc[group_name]["n_cells"] += n_cells
                for field in delay_fields:
                    val = float(group_metrics[field])
                    if math.isfinite(val):
                        delay_acc[group_name][field] += float(val) * float(n_cells)

            mask_np, t_sec, ev_sec = compute_time_mask_tsec_zero_at_R(
                meta=batch["meta"],
                T=int(target.shape[1]),
                sample_ignore_ms=float(run_cfg.get("sample_ignore_ms", 50.0)),
                resp_sec=float(run_cfg.get("resp_sec", 2.0)),
            )
            mask_t = tch.as_tensor(mask_np.astype(np.bool_), device=target.device)
            target_plot = target[:, mask_t, :]
            pred_plot = pred[:, mask_t, :]
            target_plot_viz = _time_bin_smooth_ctn(target_plot, fps=float(batch["meta"]["fps"]), bin_ms=float(viz_smooth_bin_ms))
            pred_plot_viz = _time_bin_smooth_ctn(pred_plot, fps=float(batch["meta"]["fps"]), bin_ms=float(viz_smooth_bin_ms))
            cond_names = list(batch["meta"]["cond_names"])
            thr = (top_heap[-1][0]) if len(top_heap) >= int(plot_n) and int(plot_n) > 0 else float("inf")

            for local_i in range(int(idx_net.numel())):
                mse_i = float(mse_per_neuron[local_i])
                if int(plot_n) <= 0:
                    break
                if len(top_heap) < int(plot_n) or mse_i < thr:
                    rec = {
                        "unit_key": str(batch["unit_key"]),
                        "npz_base": os.path.basename(str(batch["npz_path"])),
                        "local_i": int(local_i),
                        "mse": float(mse_i),
                        "t_sec": t_sec,
                        "ev_sec": ev_sec,
                        "cond_names": cond_names,
                        "target": target_plot[:, :, local_i].detach().cpu(),
                        "pred": pred_plot[:, :, local_i].detach().cpu(),
                        "target_viz": target_plot_viz[:, :, local_i].detach().cpu(),
                        "pred_viz": pred_plot_viz[:, :, local_i].detach().cpu(),
                    }
                    heap_counter += 1
                    top_heap.append((mse_i, heap_counter, rec))
                    top_heap.sort(key=lambda x: x[0])
                    if len(top_heap) > int(plot_n):
                        top_heap.pop(-1)
                    thr = (top_heap[-1][0]) if len(top_heap) >= int(plot_n) else float("inf")

    if len(per_unit_rows) == 0:
        raise ValueError("No units evaluated.")

    per_unit_csv = os.path.join(out_dir, f"eval_per_unit_{model_tag}.csv")
    with open(per_unit_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_unit_rows[0].keys()))
        writer.writeheader()
        for row in per_unit_rows:
            writer.writerow(row)

    sign_mse = {
        name: _weighted_mean_from_sum_count(acc["sum"], acc["count"])
        for name, acc in sign_acc.items()
    }
    celltype_mse = {
        name: _weighted_mean_from_sum_count(acc["sum"], acc["count"])
        for name, acc in type_acc.items()
        if int(acc["count"]) > 0
    }
    epoch_mse = {
        name: _weighted_mean_from_sum_count(acc["sum"], acc["count"])
        for name, acc in epoch_acc.items()
    }
    active_weight_sum = 0.0
    eval_weighted_psth_num = 0.0
    for name in EPOCH_NAMES:
        val = float(epoch_mse.get(name, float("nan")))
        w = float(loss_epoch_weights.get(name, 0.0))
        if w > 0.0 and math.isfinite(val):
            active_weight_sum += w
            eval_weighted_psth_num += w * val
    eval_weighted_psth = (eval_weighted_psth_num / active_weight_sum) if active_weight_sum > 0.0 else float("nan")

    delay_summary: Dict[str, Dict[str, float]] = {}
    for group_name, acc in delay_acc.items():
        n_cells = int(acc["n_cells"])
        if n_cells <= 0:
            continue
        vals = {field: float(acc[field]) / float(n_cells) for field in delay_fields}
        delay_summary[group_name] = {
            "n_cells": int(n_cells),
            "delay_std_ratio": (float(vals["delay_std_pred"]) / float(vals["delay_std_target"])) if abs(float(vals["delay_std_target"])) > 1e-12 else float("nan"),
            "delay_lr_sep_ratio": (float(vals["delay_lr_sep_pred"]) / float(vals["delay_lr_sep_target"])) if abs(float(vals["delay_lr_sep_target"])) > 1e-12 else float("nan"),
            "delay_mean_abs_ratio": (float(vals["delay_mean_abs_pred"]) / float(vals["delay_mean_abs_target"])) if abs(float(vals["delay_mean_abs_target"])) > 1e-12 else float("nan"),
            "sample_std_ratio": (float(vals["sample_std_pred"]) / float(vals["sample_std_target"])) if abs(float(vals["sample_std_target"])) > 1e-12 else float("nan"),
            "resp_std_ratio": (float(vals["resp_std_pred"]) / float(vals["resp_std_target"])) if abs(float(vals["resp_std_target"])) > 1e-12 else float("nan"),
            "delay_lr_sep_target": float(vals["delay_lr_sep_target"]),
            "delay_lr_sep_pred": float(vals["delay_lr_sep_pred"]),
        }

    per_celltype_csv = os.path.join(out_dir, f"eval_per_celltype_{model_tag}.csv")
    with open(per_celltype_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["celltype", "n_points", "mse"])
        writer.writeheader()
        for name in block_info_full["type_names"]:
            acc = type_acc[str(name)]
            if int(acc["count"]) <= 0:
                continue
            writer.writerow(
                {
                    "celltype": str(name),
                    "n_points": int(acc["count"]),
                    "mse": float(celltype_mse[str(name)]),
                }
            )

    losses = np.asarray([float(r["loss_mse"]) for r in per_unit_rows], dtype=float)
    eval_summary = {
        "model": str(model_path),
        "registry_dir": str(registry_dir_override or run_cfg.get("registry_dir")),
        "animal": str(animal_override or run_cfg.get("animal")),
        "model_type": str(run_cfg.get("model_type", "unknown")),
        "psth_target_source": str(psth_target_source),
        "n_units": int(len(per_unit_rows)),
        "n_total": int(shared["N_total"]),
        "n_obs_total": int(shared["N_obs_total"]),
        "n_exc_virtual": int(run_cfg.get("n_exc_virtual", 0)),
        "type_names": list(block_info_full["type_names"]),
        "full_type_counts": dict(block_info_full["full_type_counts"]),
        "block_list": list(block_info_full["block_list"]),
        "loss_mode": str(run_cfg.get("loss_mode", "mean")),
        "loss_epoch_weights": dict(loss_epoch_weights),
        "eval_psth": float(np.mean(losses)),
        "eval_weighted_psth": float(eval_weighted_psth),
        "eval_psth_median_unit": float(np.median(losses)),
        "per_sign_mse": sign_mse,
        "per_celltype_mse": celltype_mse,
        "epoch_mse": epoch_mse,
        "delay_ratios": delay_summary,
        "delay_std_ratio": float(delay_summary.get("all", {}).get("delay_std_ratio", float("nan"))),
        "delay_lr_sep_ratio": float(delay_summary.get("all", {}).get("delay_lr_sep_ratio", float("nan"))),
        "delay_lr_sep_target": float(delay_summary.get("all", {}).get("delay_lr_sep_target", float("nan"))),
        "delay_lr_sep_pred": float(delay_summary.get("all", {}).get("delay_lr_sep_pred", float("nan"))),
        "per_unit_csv": str(per_unit_csv),
        "per_celltype_csv": str(per_celltype_csv),
        "psth_smoothing": str(run_cfg.get("psth_smoothing", DEFAULT_PSTH_SMOOTHING)),
        "psth_smoothing_kernel_frames": int(psth_kernel_frames),
        "viz_smooth_bin_ms": float(viz_smooth_bin_ms),
        "viz_smooth_kernel_frames": int(viz_kernel_frames),
    }
    diag_json = save_blockrank_diagnostics(
        model=net,
        out_dir=diag_out_dir,
        stem=f"blockrank_eval_diag_{model_tag}",
        extra={
            "eval_psth": float(eval_summary["eval_psth"]),
            "eval_weighted_psth": float(eval_summary["eval_weighted_psth"]),
            "delay_std_ratio": float(eval_summary["delay_std_ratio"]),
            "delay_lr_sep_ratio": float(eval_summary["delay_lr_sep_ratio"]),
            "type_names": list(block_info_full["type_names"]),
            "full_type_counts": dict(block_info_full["full_type_counts"]),
        },
    )

    picked = [x[2] for x in top_heap]
    picked.sort(key=lambda r: float(r["mse"]))
    if len(picked) > 0:
        picked_aligned, common_t = _align_picked_records_for_mosaic(picked)
        if len(picked_aligned) > 0 and common_t is not None and int(common_t.shape[0]) > 0:
            idx_neurons = list(range(len(picked_aligned)))
            target_stack = tch.stack([rec["target"] for rec in picked_aligned], dim=2)
            pred_stack = tch.stack([rec["pred"] for rec in picked_aligned], dim=2)
            mse_vec = np.asarray([float(rec["mse"]) for rec in picked_aligned], dtype=float)
            out_png = os.path.join(out_dir, f"psth_eval_all_best_mosaic_{model_tag}_R0.png")
            plot_psth_comparison_R0_with_events(
                t_sec=np.asarray(common_t, dtype=float),
                psth_used=target_stack,
                rates_used=pred_stack,
                idx_neurons=idx_neurons,
                mse_per_neuron=mse_vec,
                cond_names=list(picked_aligned[0]["cond_names"]),
                out_path=out_png,
                ncols=max(1, int(plot_cols)),
                title=f"{model_tag} best {len(picked_aligned)} neurons (R=0)",
                event_sec=picked_aligned[0]["ev_sec"],
                show_event_labels=True,
            )
            eval_summary["mosaic_png"] = str(out_png)
            if "target_viz" in picked_aligned[0] and "pred_viz" in picked_aligned[0]:
                target_viz_stack = tch.stack([rec["target_viz"] for rec in picked_aligned], dim=2)
                pred_viz_stack = tch.stack([rec["pred_viz"] for rec in picked_aligned], dim=2)
                out_png_viz = os.path.join(out_dir, f"psth_eval_all_best_mosaic_{model_tag}_R0_vizsmooth.png")
                plot_psth_comparison_R0_with_events(
                    t_sec=np.asarray(common_t, dtype=float),
                    psth_used=target_viz_stack,
                    rates_used=pred_viz_stack,
                    idx_neurons=idx_neurons,
                    mse_per_neuron=mse_vec,
                    cond_names=list(picked_aligned[0]["cond_names"]),
                    out_path=out_png_viz,
                    ncols=max(1, int(plot_cols)),
                    title=f"{model_tag} best {len(picked_aligned)} neurons (R=0, viz smooth)",
                    event_sec=picked_aligned[0]["ev_sec"],
                    show_event_labels=True,
                )
                eval_summary["mosaic_vizsmooth_png"] = str(out_png_viz)
            else:
                eval_summary["mosaic_vizsmooth_png"] = None
        else:
            print("[WARN] Skipped mosaic plotting because no common R-aligned time grid was shared across picked neurons.", flush=True)
            eval_summary["mosaic_png"] = None
            eval_summary["mosaic_vizsmooth_png"] = None
    else:
        eval_summary["mosaic_png"] = None
        eval_summary["mosaic_vizsmooth_png"] = None

    summary_json = os.path.join(out_dir, f"eval_summary_{model_tag}.json")
    with open(summary_json, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"[OK] Saved per-unit stats -> {per_unit_csv}", flush=True)
    print(f"[OK] Saved per-celltype stats -> {per_celltype_csv}", flush=True)
    print(f"[OK] Saved eval summary -> {summary_json}", flush=True)
    print(f"[OK] Saved diagnostics -> {diag_json}", flush=True)
    print(
        f"[INFO] eval_psth={eval_summary['eval_psth']:.6f} "
        f"delay_std_ratio={eval_summary['delay_std_ratio']:.6f} "
        f"delay_lr_sep_ratio={eval_summary['delay_lr_sep_ratio']:.6f}",
        flush=True,
    )
    return eval_summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to block-rank1-dale checkpoint (.pt or .best.pt).")
    ap.add_argument("--params", default=None, help="Optional explicit params/config json.")
    ap.add_argument("--registry_dir", default=None, help="Optional override for saved registry_dir.")
    ap.add_argument("--animal", default=None, help="Optional override for saved animal.")
    ap.add_argument("--out_dir", default=None, help="Optional eval output directory.")
    ap.add_argument("--plot_n", type=int, default=72)
    ap.add_argument("--plot_cols", type=int, default=6)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    save_dir = args.out_dir or (os.path.dirname(args.model) or ".")
    eval_all_units_blockrank1_dale(
        model_path=str(args.model),
        params_path=args.params,
        registry_dir_override=args.registry_dir,
        animal_override=args.animal,
        out_dir=str(save_dir),
        plot_n=int(args.plot_n),
        plot_cols=int(args.plot_cols),
    )


if __name__ == "__main__":
    main()
