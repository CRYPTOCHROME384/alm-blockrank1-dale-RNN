import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch as tch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from losses import LossAverageTrials
from models import CellTypeBlockRank1DaleCurrentRNN
from training_current import (
    _kernel_size_from_bin_ms,
    _load_default_parameters,
    _preload_units_from_registry,
    _save_training_history_artifacts,
)
from delay_dynamics import compute_delay_shape_loss_bundle
from blockrank_dale_utils import (
    build_epoch_masks,
    build_by_unit_from_registry_dataframe,
    build_full_block_type_info,
    infer_block_type_info_from_registry_dataframe,
    load_registry_dataframe,
    save_blockrank_diagnostics,
    subset_and_reindex_registry_dataframe,
)


EPOCH_NAMES = ("sample", "delay", "response")
DEFAULT_LOSS_MODE = "epoch_weighted_mean"
DEFAULT_LOSS_EPOCH_WEIGHTS = {"sample": 1.0, "delay": 2.0, "response": 1.0}
DEFAULT_DIAG_SUBDIR_TRAIN = os.path.join("diagnostics", "train")
DEFAULT_DIAG_SUBDIR_EVAL = os.path.join("diagnostics", "eval")
DEFAULT_PSTH_SMOOTHING = "boxcar"
DEFAULT_VIZ_SMOOTH_BIN_MS = 400.0
DEFAULT_DELAY_SHAPE_LOSS_TYPE = "none"
DEFAULT_DELAY_SHAPE_GROUP_MODE = "all"
DEFAULT_DELAY_SHAPE_EPS = 1e-8
DEFAULT_DELAY_SHAPE_MIN_SCALE = 1e-6


def _atomic_torch_save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    tch.save(obj, tmp)
    os.replace(tmp, path)


def _count_trainable_parameters(net) -> int:
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def _resolve_aux_dir(base_dir: str, subdir: Optional[str]) -> str:
    if subdir in (None, ""):
        return str(base_dir)
    subdir_str = str(subdir)
    return subdir_str if os.path.isabs(subdir_str) else os.path.join(str(base_dir), subdir_str)


def _parse_loss_mode(loss_mode: Optional[str]) -> str:
    mode = str(loss_mode or "mean").strip().lower()
    aliases = {
        "mean": "mean",
        "global_mean": "mean",
        "legacy_mean": "mean",
        "epoch_weighted_mean": "epoch_weighted_mean",
    }
    if mode not in aliases:
        raise ValueError(f"Unsupported loss_mode={loss_mode!r}; expected one of {sorted(aliases)}")
    return aliases[mode]


def _normalize_loss_epoch_weights(loss_epoch_weights: Optional[Any]) -> Dict[str, float]:
    if loss_epoch_weights is None:
        src = dict(DEFAULT_LOSS_EPOCH_WEIGHTS)
    elif isinstance(loss_epoch_weights, dict):
        src = {str(k): float(v) for k, v in loss_epoch_weights.items()}
    elif isinstance(loss_epoch_weights, (list, tuple)) and len(loss_epoch_weights) == len(EPOCH_NAMES):
        src = {name: float(val) for name, val in zip(EPOCH_NAMES, loss_epoch_weights)}
    else:
        raise TypeError(
            "loss_epoch_weights must be a dict with keys sample/delay/response or a length-3 sequence."
        )

    out: Dict[str, float] = {}
    for name in EPOCH_NAMES:
        val = float(src.get(name, DEFAULT_LOSS_EPOCH_WEIGHTS[name]))
        if not np.isfinite(val) or val < 0.0:
            raise ValueError(f"loss_epoch_weights[{name!r}] must be finite and >=0, got {val!r}")
        out[name] = val
    if not any(v > 0.0 for v in out.values()):
        raise ValueError("loss_epoch_weights must contain at least one positive weight.")
    return out


def _masked_loss(loss_fn, target: tch.Tensor, pred: tch.Tensor, mask_t: tch.Tensor) -> tch.Tensor:
    if mask_t.dtype != tch.bool:
        mask_t = mask_t.bool()
    if int(mask_t.sum().item()) <= 0:
        return pred.new_full((), float("nan"))
    return loss_fn(target[:, mask_t, :], pred[:, mask_t, :])


def _compute_epoch_loss_bundle(
    *,
    loss_fn,
    target: tch.Tensor,
    pred: tch.Tensor,
    loss_mask_t: tch.Tensor,
    epoch_masks: Dict[str, np.ndarray],
    device: tch.device,
    loss_mode: str,
    loss_epoch_weights: Dict[str, float],
) -> Dict[str, Any]:
    overall_loss = _masked_loss(loss_fn, target, pred, loss_mask_t)
    epoch_losses: Dict[str, tch.Tensor] = {}
    weighted_terms: List[tch.Tensor] = []
    active_weights: List[float] = []

    for name in EPOCH_NAMES:
        epoch_mask_t = tch.as_tensor(np.asarray(epoch_masks[name], dtype=np.bool_), device=device)
        loss_epoch = _masked_loss(loss_fn, target, pred, epoch_mask_t)
        epoch_losses[name] = loss_epoch
        weight = float(loss_epoch_weights.get(name, 0.0))
        if weight > 0.0 and bool(tch.isfinite(loss_epoch).item()):
            weighted_terms.append(loss_epoch * float(weight))
            active_weights.append(float(weight))

    if len(weighted_terms) > 0:
        weighted_loss = tch.stack(weighted_terms).sum() / float(sum(active_weights))
    else:
        weighted_loss = overall_loss

    return {
        "overall_loss": overall_loss,
        "weighted_loss": weighted_loss,
        "selected_loss": weighted_loss if str(loss_mode) == "epoch_weighted_mean" else overall_loss,
        "epoch_losses": epoch_losses,
    }


def _configure_runtime_device(
    *,
    device: str,
    require_cuda: bool,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    matmul_precision: str,
) -> tuple[tch.device, Dict[str, Any]]:
    requested = str(device or "auto").strip().lower()
    cuda_available = bool(tch.cuda.is_available())

    if requested in {"", "auto"}:
        resolved = "cuda" if cuda_available else "cpu"
    elif requested == "gpu":
        resolved = "cuda"
    else:
        resolved = str(device)

    wants_cuda = str(resolved).startswith("cuda")
    if wants_cuda and (not cuda_available):
        msg = (
            f"Requested device={device!r}, but torch.cuda.is_available() is False. "
            "Training would fall back to CPU."
        )
        if bool(require_cuda) or requested not in {"", "auto"}:
            raise RuntimeError(msg)
        resolved = "cpu"

    if str(resolved).startswith("cuda") and (":" not in str(resolved)):
        resolved = "cuda:0"

    dev = tch.device(str(resolved))

    if hasattr(tch, "set_float32_matmul_precision"):
        try:
            tch.set_float32_matmul_precision(str(matmul_precision))
        except Exception:
            pass

    runtime_info: Dict[str, Any] = {
        "requested_device": str(device),
        "resolved_device": str(dev),
        "cuda_available": bool(cuda_available),
        "require_cuda": bool(require_cuda),
        "enable_tf32": bool(enable_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "matmul_precision": str(matmul_precision),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", None),
    }

    if dev.type == "cuda":
        target_idx = 0 if dev.index is None else int(dev.index)
        tch.cuda.set_device(target_idx)
        if hasattr(tch.backends, "cuda") and hasattr(tch.backends.cuda, "matmul"):
            tch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        if hasattr(tch.backends, "cudnn"):
            tch.backends.cudnn.allow_tf32 = bool(enable_tf32)
            tch.backends.cudnn.benchmark = bool(cudnn_benchmark)
        runtime_info["cuda_device_name"] = str(tch.cuda.get_device_name(target_idx))
        runtime_info["cuda_device_index"] = int(tch.cuda.current_device())
    else:
        runtime_info["cuda_device_name"] = None
        runtime_info["cuda_device_index"] = None

    return dev, runtime_info


def _select_psth_target(batch: Dict[str, Any], psth_target_source: str) -> tch.Tensor:
    source = str(psth_target_source)
    if source == "stage1_psth":
        return batch["psth_sub"]
    if source == "trials_mean":
        if "trials_mean_psth" not in batch:
            raise KeyError("[TRALIGN] unit missing trials_mean_psth while psth_target_source='trials_mean'")
        return batch["trials_mean_psth"]
    raise ValueError(f"Unsupported psth_target_source={psth_target_source!r}")


def _local_delay_shape_group_args(
    *,
    idx_net: tch.Tensor,
    block_info_full: Dict[str, Any],
    group_mode: str,
) -> Dict[str, Any]:
    mode = str(group_mode or DEFAULT_DELAY_SHAPE_GROUP_MODE).strip().lower()
    if mode == "all":
        return {"group_mode": "all", "group_index": None, "group_names": None}
    if mode != "celltype":
        raise ValueError(f"Unsupported delay_shape_group_mode={group_mode!r}; expected 'all' or 'celltype'")
    idx_np = idx_net.detach().cpu().numpy().astype(np.int64)
    full_type_index = np.asarray(block_info_full["full_type_index"], dtype=np.int64)
    local_type_index = full_type_index[idx_np]
    return {
        "group_mode": "celltype",
        "group_index": local_type_index.tolist(),
        "group_names": list(block_info_full["type_names"]),
    }


def _global_eval_blockrank(
    *,
    net,
    units: List[Dict[str, Any]],
    loss_fn,
    psth_target_source: str,
    sample_ignore_ms: float,
    resp_sec: float,
    loss_mode: str,
    loss_epoch_weights: Dict[str, float],
) -> Dict[str, float]:
    net.eval()
    sum_total = 0.0
    sum_psth = 0.0
    n_units = 0
    epoch_acc = {name: {"sum": 0.0, "count": 0} for name in EPOCH_NAMES}

    with tch.no_grad():
        for batch in units:
            u = tch.nan_to_num(batch["u"], nan=0.0, posinf=0.0, neginf=0.0)
            idx_net = batch["idx_net"]
            time_mask = batch["time_mask"]
            target = tch.nan_to_num(_select_psth_target(batch, psth_target_source), nan=0.0, posinf=0.0, neginf=0.0)

            out = net(u, h0=None, noise_std=0.0, return_rate=True)
            pred_sub = out["rate"].index_select(dim=2, index=idx_net)
            epoch_masks = build_epoch_masks(
                batch["meta"],
                T=int(target.shape[1]),
                sample_ignore_ms=float(sample_ignore_ms),
                resp_sec=float(resp_sec),
            )
            loss_bundle = _compute_epoch_loss_bundle(
                loss_fn=loss_fn,
                target=target,
                pred=pred_sub,
                loss_mask_t=time_mask,
                epoch_masks=epoch_masks,
                device=target.device,
                loss_mode=str(loss_mode),
                loss_epoch_weights=loss_epoch_weights,
            )
            loss_psth = loss_bundle["overall_loss"]
            loss_total = loss_bundle["selected_loss"]
            sum_psth += float(loss_psth.detach().cpu().item())
            sum_total += float(loss_total.detach().cpu().item())
            for epoch_name in EPOCH_NAMES:
                epoch_mask_t = tch.as_tensor(np.asarray(epoch_masks[epoch_name], dtype=np.bool_), device=target.device)
                if int(epoch_mask_t.sum().item()) <= 0:
                    continue
                diff_epoch = (pred_sub[:, epoch_mask_t, :] - target[:, epoch_mask_t, :]).pow(2)
                epoch_acc[epoch_name]["sum"] += float(diff_epoch.sum().detach().cpu().item())
                epoch_acc[epoch_name]["count"] += int(diff_epoch.numel())
            n_units += 1

    rec_stats = net.recurrent_regularization_stats()
    denom = max(int(n_units), 1)
    epoch_mse = {
        name: (float(epoch_acc[name]["sum"]) / float(epoch_acc[name]["count"]))
        if int(epoch_acc[name]["count"]) > 0
        else float("nan")
        for name in EPOCH_NAMES
    }
    return {
        "total": sum_total / float(denom),
        "psth": sum_psth / float(denom),
        "psth_sample": float(epoch_mse["sample"]),
        "psth_delay": float(epoch_mse["delay"]),
        "psth_response": float(epoch_mse["response"]),
        "J": float(rec_stats["J_mean_sq"]),
        "J_frob": float(rec_stats["J_frob"]),
        "J_maxabs": float(rec_stats["J_maxabs"]),
        "n_units": int(n_units),
    }


def train_current_alm_global_blockrank1_dale(
    registry_dir: str,
    animal: str,
    out_dir: str,
    *,
    device: str = "auto",
    require_cuda: bool = False,
    enable_tf32: bool = True,
    cudnn_benchmark: bool = True,
    matmul_precision: str = "high",
    max_epochs: int = 5000,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    seed: int = 42,
    grad_clip: Optional[float] = 1.0,
    print_every: int = 50,
    dt: float = 0.03436,
    tau: float = 0.01,
    substeps: int = 6,
    nonlinearity: str = "tanh",
    n_exc_virtual: int = 0,
    unit_sampling: str = "random",
    max_sessions: Optional[int] = None,
    cond_filter=None,
    max_time=None,
    psth_bin_ms: float = 200.0,
    sample_ignore_ms: float = 50.0,
    resp_sec: float = 2.0,
    save_best_every: int = 100,
    save_latest_every: int = 0,
    eval_every: int = 100,
    best_metric: str = "eval_psth",
    loss_mode: str = "mean",
    loss_epoch_weights: Optional[Any] = None,
    use_trials: bool = True,
    strict_trials_align: bool = True,
    trials_root: Optional[str] = None,
    trials_path_mode: str = "auto_from_stage1",
    trial_keys: Optional[List[str]] = None,
    debug_trials_align: bool = False,
    model_type: str = "celltype_block_rank1_dale",
    psth_target_source: str = "trials_mean",
    block_rank: int = 1,
    dale_strict: bool = True,
    factor_nonlinearity: str = "softplus",
    normalize_uv: bool = True,
    A_nonlinearity: str = "softplus",
    A_l2: float = 0.0,
    uv_l2: float = 0.0,
    celltype_mode: str = "broadE_inh_subclass",
    block_celltype_min_count: int = 10,
    block_registry_label_cols: Optional[List[str]] = None,
    other_inh_label: str = "I_other",
    init_A: float = 0.10,
    init_factor_scale: float = 0.02,
    eps: float = 1e-8,
    use_x0_noise: bool = False,
    use_process_noise: bool = False,
    lambda_var: float = 0.0,
    lambda_celltype: float = 0.0,
    diag_every: int = 1000,
    diag_subdir_train: str = DEFAULT_DIAG_SUBDIR_TRAIN,
    diag_subdir_eval: str = DEFAULT_DIAG_SUBDIR_EVAL,
    viz_smooth_bin_ms: float = DEFAULT_VIZ_SMOOTH_BIN_MS,
    delay_shape_loss_type: str = DEFAULT_DELAY_SHAPE_LOSS_TYPE,
    delay_shape_lambda: float = 0.0,
    delay_shape_eps: float = DEFAULT_DELAY_SHAPE_EPS,
    delay_shape_min_scale: float = DEFAULT_DELAY_SHAPE_MIN_SCALE,
    delay_shape_group_mode: str = DEFAULT_DELAY_SHAPE_GROUP_MODE,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if str(model_type) != "celltype_block_rank1_dale":
        raise ValueError(f"model_type must be 'celltype_block_rank1_dale', got {model_type!r}")
    if str(psth_target_source) not in {"stage1_psth", "trials_mean"}:
        raise ValueError(f"Unsupported psth_target_source={psth_target_source!r}")
    if int(block_rank) != 1:
        raise ValueError(f"Only block_rank=1 is supported, got {block_rank}")
    if not bool(dale_strict):
        raise ValueError("This implementation requires dale_strict=True.")
    if bool(use_x0_noise) or bool(use_process_noise):
        raise ValueError("First block-rank1 version is deterministic only: use_x0_noise/use_process_noise must be false.")
    if float(lambda_var) != 0.0:
        raise ValueError("First block-rank1 version does not support variance loss; lambda_var must be 0.")
    if float(lambda_celltype) != 0.0:
        raise ValueError("First block-rank1 version does not support celltype template loss; lambda_celltype must be 0.")
    if not bool(use_trials) and str(psth_target_source) == "trials_mean":
        raise ValueError("psth_target_source='trials_mean' requires use_trials=true.")
    loss_mode_norm = _parse_loss_mode(loss_mode)
    loss_epoch_weights_norm = _normalize_loss_epoch_weights(loss_epoch_weights)
    delay_shape_loss_type = str(delay_shape_loss_type or DEFAULT_DELAY_SHAPE_LOSS_TYPE)
    delay_shape_group_mode = str(delay_shape_group_mode or DEFAULT_DELAY_SHAPE_GROUP_MODE)
    if float(delay_shape_lambda) < 0.0:
        raise ValueError(f"delay_shape_lambda must be >=0, got {delay_shape_lambda}")

    os.makedirs(out_dir, exist_ok=True)

    try:
        default_parameters = _load_default_parameters(None)
    except Exception:
        default_parameters = {}

    requested_device = str(device if device not in (None, "") else default_parameters.get("device", "auto"))
    dev, runtime_info = _configure_runtime_device(
        device=requested_device,
        require_cuda=bool(require_cuda),
        enable_tf32=bool(enable_tf32),
        cudnn_benchmark=bool(cudnn_benchmark),
        matmul_precision=str(matmul_precision),
    )

    rng = np.random.RandomState(int(seed))
    np.random.seed(int(seed))
    tch.manual_seed(int(seed))
    if dev.type == "cuda":
        tch.cuda.manual_seed_all(int(seed))

    registry_df_full, registry_csv = load_registry_dataframe(registry_dir=str(registry_dir), animal=str(animal))
    registry_df, selected_unit_keys, reindex_map = subset_and_reindex_registry_dataframe(
        registry_df_full,
        max_sessions=max_sessions,
    )
    by_unit = build_by_unit_from_registry_dataframe(registry_df)

    units, shared = _preload_units_from_registry(
        by_unit=by_unit,
        n_exc_virtual=int(n_exc_virtual),
        device=dev,
        cond_filter=cond_filter,
        max_time=max_time,
        psth_bin_ms=float(psth_bin_ms),
        sample_ignore_ms=float(sample_ignore_ms),
        resp_sec=float(resp_sec),
        use_trials=bool(use_trials),
        strict_trials_align=bool(strict_trials_align),
        trials_root=trials_root,
        trials_path_mode=str(trials_path_mode),
        trial_keys=trial_keys,
        debug_trials_align=bool(debug_trials_align),
        phase3_precompute_var_real=False,
        var_unbiased=False,
        min_trials_for_var_real=3,
    )
    if len(units) == 0:
        raise RuntimeError("No units were preloaded from registry.")

    block_info_obs = infer_block_type_info_from_registry_dataframe(
        registry_df,
        celltype_mode=str(celltype_mode),
        registry_label_cols=block_registry_label_cols,
        inh_min_count=int(block_celltype_min_count),
        other_inh_label=str(other_inh_label),
    )
    block_info_full = build_full_block_type_info(block_info_obs, n_exc_virtual=int(n_exc_virtual))

    n_total = int(shared["N_total"])
    D_in = int(units[0]["u"].shape[-1])
    diag_train_dir = _resolve_aux_dir(out_dir, diag_subdir_train)
    diag_eval_dir = _resolve_aux_dir(out_dir, diag_subdir_eval)
    psth_kernel_frames = int(_kernel_size_from_bin_ms(float(shared["fps"]), float(psth_bin_ms)))
    viz_smooth_bin_ms = max(float(viz_smooth_bin_ms), float(psth_bin_ms))
    viz_kernel_frames = int(_kernel_size_from_bin_ms(float(shared["fps"]), float(viz_smooth_bin_ms)))
    if int(len(block_info_full["full_type_index"])) != n_total:
        raise ValueError(
            "full_type_index length mismatch after registry preprocessing: "
            f"{len(block_info_full['full_type_index'])} vs N_total={n_total}"
        )

    net = CellTypeBlockRank1DaleCurrentRNN(
        N=int(n_total),
        D_in=int(D_in),
        neuron_type_index=tch.as_tensor(block_info_full["full_type_index"], dtype=tch.long),
        type_names=list(block_info_full["type_names"]),
        type_signs=list(block_info_full["type_signs"]),
        dt=float(dt),
        tau=float(tau),
        substeps=int(substeps),
        nonlinearity=str(nonlinearity),
        block_rank=int(block_rank),
        factor_nonlinearity=str(factor_nonlinearity),
        A_nonlinearity=str(A_nonlinearity),
        normalize_uv=bool(normalize_uv),
        eps=float(eps),
        init_A=float(init_A),
        init_factor_scale=float(init_factor_scale),
        device=dev,
    ).to(dev)

    loss_fn = LossAverageTrials()
    opt = tch.optim.Adam(net.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    tag = f"{animal}_global_nobs{shared['N_obs_total']}_nexc{int(n_exc_virtual)}_ntotal{n_total}"
    model_path = os.path.join(out_dir, f"rnn_current_{tag}.pt")
    best_path = os.path.join(out_dir, f"rnn_current_{tag}.best.pt")
    latest_path = os.path.join(out_dir, f"rnn_current_{tag}.latest.pt")
    meta_path = os.path.join(out_dir, f"rnn_current_{tag}.meta.json")
    params_path = os.path.join(out_dir, f"rnn_current_{tag}.train_config.json")
    history_title = f"{animal} blockrank1_dale training history"

    run_cfg_to_save: Dict[str, Any] = {
        "registry_dir": str(registry_dir),
        "registry_csv": str(registry_csv),
        "animal": str(animal),
        "out_dir": str(out_dir),
        "device": str(requested_device),
        "resolved_device": str(dev),
        "require_cuda": bool(require_cuda),
        "enable_tf32": bool(enable_tf32),
        "cudnn_benchmark": bool(cudnn_benchmark),
        "matmul_precision": str(matmul_precision),
        "max_epochs": int(max_epochs),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "seed": int(seed),
        "grad_clip": None if grad_clip is None else float(grad_clip),
        "print_every": int(print_every),
        "dt": float(dt),
        "tau": float(tau),
        "substeps": int(substeps),
        "nonlinearity": str(nonlinearity),
        "n_exc_virtual": int(n_exc_virtual),
        "unit_sampling": str(unit_sampling),
        "max_sessions": None if max_sessions is None else int(max_sessions),
        "cond_filter": None if cond_filter is None else [str(x) for x in cond_filter],
        "max_time": None if max_time is None else int(max_time),
        "psth_bin_ms": float(psth_bin_ms),
        "sample_ignore_ms": float(sample_ignore_ms),
        "resp_sec": float(resp_sec),
        "save_best_every": int(save_best_every),
        "save_latest_every": int(save_latest_every),
        "eval_every": int(eval_every),
        "best_metric": str(best_metric),
        "loss_mode": str(loss_mode_norm),
        "loss_epoch_weights": dict(loss_epoch_weights_norm),
        "use_trials": bool(use_trials),
        "strict_trials_align": bool(strict_trials_align),
        "trials_root": None if trials_root is None else str(trials_root),
        "trials_path_mode": str(trials_path_mode),
        "trial_keys": None if trial_keys is None else [str(x) for x in trial_keys],
        "debug_trials_align": bool(debug_trials_align),
        "model_type": str(model_type),
        "psth_target_source": str(psth_target_source),
        "block_rank": int(block_rank),
        "dale_strict": bool(dale_strict),
        "factor_nonlinearity": str(factor_nonlinearity),
        "normalize_uv": bool(normalize_uv),
        "A_nonlinearity": str(A_nonlinearity),
        "A_l2": float(A_l2),
        "uv_l2": float(uv_l2),
        "celltype_mode": str(celltype_mode),
        "block_celltype_min_count": int(block_celltype_min_count),
        "block_registry_label_cols": None if block_registry_label_cols is None else [str(x) for x in block_registry_label_cols],
        "other_inh_label": str(other_inh_label),
        "init_A": float(init_A),
        "init_factor_scale": float(init_factor_scale),
        "eps": float(eps),
        "psth_smoothing": DEFAULT_PSTH_SMOOTHING,
        "psth_smoothing_kernel_frames": int(psth_kernel_frames),
        "viz_smooth_bin_ms": float(viz_smooth_bin_ms),
        "viz_smooth_kernel_frames": int(viz_kernel_frames),
        "diag_every": int(diag_every),
        "diag_subdir_train": str(diag_subdir_train),
        "diag_subdir_eval": str(diag_subdir_eval),
        "delay_shape_loss_type": str(delay_shape_loss_type),
        "delay_shape_lambda": float(delay_shape_lambda),
        "delay_shape_eps": float(delay_shape_eps),
        "delay_shape_min_scale": float(delay_shape_min_scale),
        "delay_shape_group_mode": str(delay_shape_group_mode),
        "use_x0_noise": bool(use_x0_noise),
        "use_process_noise": bool(use_process_noise),
        "lambda_var": float(lambda_var),
        "lambda_celltype": float(lambda_celltype),
        "selected_unit_keys": list(selected_unit_keys),
        "global_idx_reindexed": True,
    }
    if isinstance(run_config, dict):
        merged = dict(run_config)
        merged["_resolved_runtime"] = dict(run_cfg_to_save)
        run_cfg_to_save = merged
    with open(params_path, "w") as f:
        json.dump(run_cfg_to_save, f, indent=2)

    n_trainable_params = _count_trainable_parameters(net)
    n_recurrent_trainable_params = int(net.recurrent_trainable_parameter_count())
    n_dense_equivalent = int(net.equivalent_full_rank_parameter_count())

    print(f"[INFO] model_type={model_type}", flush=True)
    print(
        f"[INFO] device requested={runtime_info['requested_device']} resolved={runtime_info['resolved_device']} "
        f"cuda_available={runtime_info['cuda_available']} require_cuda={runtime_info['require_cuda']} "
        f"CUDA_VISIBLE_DEVICES={runtime_info['cuda_visible_devices']}",
        flush=True,
    )
    if runtime_info.get("cuda_device_name", None) is not None:
        print(
            f"[INFO] cuda_device_index={runtime_info['cuda_device_index']} "
            f"cuda_device_name={runtime_info['cuda_device_name']} "
            f"tf32={runtime_info['enable_tf32']} cudnn_benchmark={runtime_info['cudnn_benchmark']} "
            f"matmul_precision={runtime_info['matmul_precision']}",
            flush=True,
        )
    print(f"[INFO] psth_target_source={psth_target_source}", flush=True)
    print(
        f"[INFO] psth_smoothing={DEFAULT_PSTH_SMOOTHING} psth_bin_ms={float(psth_bin_ms):.1f} "
        f"psth_kernel_frames={int(psth_kernel_frames)} viz_smooth_bin_ms={float(viz_smooth_bin_ms):.1f} "
        f"viz_kernel_frames={int(viz_kernel_frames)}",
        flush=True,
    )
    print(
        f"[INFO] loss_mode={loss_mode_norm} loss_epoch_weights={loss_epoch_weights_norm} "
        f"delay_shape_loss_type={delay_shape_loss_type} delay_shape_lambda={float(delay_shape_lambda):.6g} "
        f"delay_shape_group_mode={delay_shape_group_mode} delay_shape_eps={float(delay_shape_eps):.3g} "
        f"delay_shape_min_scale={float(delay_shape_min_scale):.3g} "
        f"diag_every={int(diag_every)} diag_train_dir={diag_train_dir} diag_eval_dir={diag_eval_dir}",
        flush=True,
    )
    print(
        f"[INFO] preloaded_units={len(units)} shared(C={shared['C']},T={shared['T']},fps={shared['fps']:.3f}) "
        f"total_neuron_count={n_total} observed_neurons={shared['N_obs_total']} virtual_exc={int(n_exc_virtual)}",
        flush=True,
    )
    print(f"[INFO] celltype_mode={celltype_mode} registry_label_col={block_info_obs['registry_label_col_used']}", flush=True)
    print(f"[INFO] observed_cell_type_counts={block_info_obs['observed_type_counts']}", flush=True)
    print(f"[INFO] full_cell_type_counts={block_info_full['full_type_counts']}", flush=True)
    print(f"[INFO] block_list={block_info_full['block_list']}", flush=True)
    print(f"[INFO] presyn_N_b={block_info_full['presyn_count_by_type']}", flush=True)
    print(
        f"[INFO] recurrent_trainable_params={n_recurrent_trainable_params} "
        f"equivalent_fullrank_params={n_dense_equivalent} total_trainable_params={n_trainable_params}",
        flush=True,
    )
    for note in block_info_obs.get("notes", []):
        print(f"[INFO] block_celltype_note={note}", flush=True)

    diag_path0 = save_blockrank_diagnostics(
        model=net,
        out_dir=diag_train_dir,
        stem="blockrank_diag_ep000000",
        extra={
            "epoch": 0,
            "model_type": str(model_type),
            "psth_target_source": str(psth_target_source),
            "loss_mode": str(loss_mode_norm),
            "loss_epoch_weights": dict(loss_epoch_weights_norm),
            "type_names": list(block_info_full["type_names"]),
            "full_type_counts": dict(block_info_full["full_type_counts"]),
        },
    )
    print(f"[OK] Saved initial diagnostics -> {diag_path0}", flush=True)

    valid_best_metrics = {"train_step_total", "eval_total", "eval_psth"}
    if str(best_metric) not in valid_best_metrics:
        raise ValueError(f"best_metric must be one of {sorted(valid_best_metrics)}, got {best_metric!r}")

    best_total = float("inf")
    best_score = float("inf")
    best_ep = 0
    best_train_ep = 0
    best_state = None
    best_eval_stats = None
    history_rows: List[Dict[str, Any]] = []
    unit_order = list(range(len(units)))
    t0 = time.time()

    def _flush_history_safe() -> None:
        if len(history_rows) == 0:
            return
        try:
            _save_training_history_artifacts(history_rows, out_dir=out_dir, title=history_title)
        except Exception as exc:
            print(f"[WARN] Failed to save training history artifacts: {exc}", flush=True)

    try:
        for ep in range(int(max_epochs)):
            net.train()
            if str(unit_sampling) == "random":
                ui = int(rng.randint(0, len(units)))
            elif str(unit_sampling) == "cycle":
                ui = unit_order[ep % len(unit_order)]
            else:
                raise ValueError(f"unit_sampling must be 'random' or 'cycle', got {unit_sampling!r}")

            batch = units[ui]
            u = tch.nan_to_num(batch["u"], nan=0.0, posinf=0.0, neginf=0.0)
            idx_net = batch["idx_net"]
            time_mask = batch["time_mask"]
            target = tch.nan_to_num(_select_psth_target(batch, psth_target_source), nan=0.0, posinf=0.0, neginf=0.0)

            if ep == 0:
                print(f"[TRDBG] psth_target_source={psth_target_source} unit={batch.get('unit_key', 'NA')}", flush=True)

            opt.zero_grad(set_to_none=True)
            out = net(u, h0=None, noise_std=0.0, return_rate=True)
            pred_sub = out["rate"].index_select(dim=2, index=idx_net)
            epoch_masks = build_epoch_masks(
                batch["meta"],
                T=int(target.shape[1]),
                sample_ignore_ms=float(sample_ignore_ms),
                resp_sec=float(resp_sec),
            )
            loss_bundle = _compute_epoch_loss_bundle(
                loss_fn=loss_fn,
                target=target,
                pred=pred_sub,
                loss_mask_t=time_mask,
                epoch_masks=epoch_masks,
                device=target.device,
                loss_mode=str(loss_mode_norm),
                loss_epoch_weights=loss_epoch_weights_norm,
            )
            loss_psth = loss_bundle["selected_loss"]
            delay_shape_group_args = _local_delay_shape_group_args(
                idx_net=idx_net,
                block_info_full=block_info_full,
                group_mode=str(delay_shape_group_mode),
            )
            delay_shape_bundle = compute_delay_shape_loss_bundle(
                target=target,
                pred=pred_sub,
                delay_mask=epoch_masks["delay"],
                loss_type=str(delay_shape_loss_type),
                eps=float(delay_shape_eps),
                min_scale=float(delay_shape_min_scale),
                group_mode=str(delay_shape_group_args["group_mode"]),
                group_index=delay_shape_group_args["group_index"],
                group_names=delay_shape_group_args["group_names"],
            )
            loss_delay_shape = delay_shape_bundle["loss"]
            weighted_delay_shape = float(delay_shape_lambda) * loss_delay_shape
            reg_block = net.recurrent_regularization_loss(A_l2=float(A_l2), uv_l2=float(uv_l2))
            loss_total = loss_psth + reg_block + weighted_delay_shape

            if not tch.isfinite(loss_total):
                raise FloatingPointError(
                    "Non-finite loss detected: "
                    + json.dumps(
                        {
                            "epoch": int(ep + 1),
                            "unit_key": str(batch.get("unit_key", "NA")),
                            "loss_total": str(loss_total.detach().cpu().item()),
                            "loss_psth": str(loss_psth.detach().cpu().item()),
                            "loss_delay_shape": str(loss_delay_shape.detach().cpu().item()),
                            "loss_reg": str(reg_block.detach().cpu().item()),
                        },
                        indent=2,
                    )
                )

            loss_total.backward()
            if grad_clip is not None and float(grad_clip) > 0.0:
                tch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=float(grad_clip))
            opt.step()

            rec_stats = net.recurrent_regularization_stats()
            total_val = float(loss_total.detach().cpu().item())
            psth_val = float(loss_bundle["overall_loss"].detach().cpu().item())
            psth_weighted_val = float(loss_bundle["weighted_loss"].detach().cpu().item())
            delay_shape_val = float(loss_delay_shape.detach().cpu().item())
            weighted_delay_shape_val = float(weighted_delay_shape.detach().cpu().item())
            epoch_train_vals = {
                name: float(loss_bundle["epoch_losses"][name].detach().cpu().item())
                if bool(tch.isfinite(loss_bundle["epoch_losses"][name]).item())
                else float("nan")
                for name in EPOCH_NAMES
            }
            reg_val = float(reg_block.detach().cpu().item())
            J_val = float(rec_stats["J_mean_sq"])
            J_frob = float(rec_stats["J_frob"])
            J_maxabs = float(rec_stats["J_maxabs"])
            eval_total = float("nan")
            eval_psth = float("nan")
            eval_J = float("nan")
            eval_psth_sample = float("nan")
            eval_psth_delay = float("nan")
            eval_psth_response = float("nan")

            if total_val < best_total:
                best_total = total_val
                best_train_ep = int(ep + 1)

            if str(best_metric) == "train_step_total" and total_val < best_score:
                best_score = total_val
                best_ep = int(ep + 1)
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

            if ep == 0 or ((ep + 1) % max(1, int(print_every)) == 0):
                elapsed = time.time() - t0
                print(
                    "[blockrank] ep=%d/%d total=%.6f psth=%.6f weighted=%.6f seg[s=%.6f d=%.6f r=%.6f] "
                    "shape[%s]=%.6f wshape=%.6f reg=%.6f "
                    "J=%.6g Jf=%.6g Jmax=%.6g best=%.6f unit=%s (K=%d) elapsed=%.1fs"
                    % (
                        ep + 1,
                        int(max_epochs),
                        total_val,
                        psth_val,
                        psth_weighted_val,
                        float(epoch_train_vals["sample"]),
                        float(epoch_train_vals["delay"]),
                        float(epoch_train_vals["response"]),
                        str(delay_shape_loss_type),
                        float(delay_shape_val),
                        float(weighted_delay_shape_val),
                        reg_val,
                        J_val,
                        J_frob,
                        J_maxabs,
                        float(best_total),
                        str(batch.get("unit_key", "NA")),
                        int(idx_net.numel()),
                        elapsed,
                    ),
                    flush=True,
                )

            do_eval = int(eval_every) > 0 and (ep == 0 or ((ep + 1) % int(eval_every) == 0))
            if do_eval:
                ev = _global_eval_blockrank(
                    net=net,
                    units=units,
                    loss_fn=loss_fn,
                    psth_target_source=str(psth_target_source),
                    sample_ignore_ms=float(sample_ignore_ms),
                    resp_sec=float(resp_sec),
                    loss_mode=str(loss_mode_norm),
                    loss_epoch_weights=loss_epoch_weights_norm,
                )
                eval_total = float(ev["total"])
                eval_psth = float(ev["psth"])
                eval_J = float(ev["J"])
                eval_psth_sample = float(ev["psth_sample"])
                eval_psth_delay = float(ev["psth_delay"])
                eval_psth_response = float(ev["psth_response"])

                sel_val = eval_psth if str(best_metric) == "eval_psth" else (eval_total if str(best_metric) == "eval_total" else None)
                if sel_val is not None and float(sel_val) < best_score:
                    best_score = float(sel_val)
                    best_ep = int(ep + 1)
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    best_eval_stats = {
                        "total": float(ev["total"]),
                        "psth": float(ev["psth"]),
                        "J": float(ev["J"]),
                        "J_frob": float(ev["J_frob"]),
                        "J_maxabs": float(ev["J_maxabs"]),
                        "psth_sample": float(ev["psth_sample"]),
                        "psth_delay": float(ev["psth_delay"]),
                        "psth_response": float(ev["psth_response"]),
                        "n_units": int(ev["n_units"]),
                    }
                    best_diag_path = save_blockrank_diagnostics(
                        model=net,
                        out_dir=diag_train_dir,
                        stem="blockrank_diag_best",
                        extra={
                            "epoch": int(ep + 1),
                            "best_metric": str(best_metric),
                            "best_score": float(best_score),
                            "eval_total": float(ev["total"]),
                            "eval_psth": float(ev["psth"]),
                            "eval_psth_sample": float(ev["psth_sample"]),
                            "eval_psth_delay": float(ev["psth_delay"]),
                            "eval_psth_response": float(ev["psth_response"]),
                            "type_names": list(block_info_full["type_names"]),
                            "full_type_counts": dict(block_info_full["full_type_counts"]),
                        },
                    )
                    print(f"[OK] Saved best diagnostics -> {best_diag_path}", flush=True)

                diag_path = None
                if int(diag_every) > 0 and ((ep + 1) % int(diag_every) == 0):
                    diag_path = save_blockrank_diagnostics(
                        model=net,
                        out_dir=diag_train_dir,
                        stem=f"blockrank_diag_ep{int(ep + 1):06d}",
                        extra={
                            "epoch": int(ep + 1),
                            "eval_total": float(ev["total"]),
                            "eval_psth": float(ev["psth"]),
                            "eval_psth_sample": float(ev["psth_sample"]),
                            "eval_psth_delay": float(ev["psth_delay"]),
                            "eval_psth_response": float(ev["psth_response"]),
                            "type_names": list(block_info_full["type_names"]),
                            "full_type_counts": dict(block_info_full["full_type_counts"]),
                        },
                    )
                print(
                    "[eval] ep=%d/%d best_metric=%s score=%.6f | eval_total=%.6f eval_psth=%.6f seg[s=%.6f d=%.6f r=%.6f] "
                    "J=%.6g Jf=%.6g Jmax=%.6g | bestSel=%.6f@ep%d"
                    % (
                        ep + 1,
                        int(max_epochs),
                        str(best_metric),
                        float("nan") if sel_val is None else float(sel_val),
                        float(ev["total"]),
                        float(ev["psth"]),
                        float(ev["psth_sample"]),
                        float(ev["psth_delay"]),
                        float(ev["psth_response"]),
                        float(ev["J"]),
                        float(ev["J_frob"]),
                        float(ev["J_maxabs"]),
                        float(best_score),
                        int(best_ep),
                    ),
                    flush=True,
                )
                if diag_path is not None:
                    print(f"[OK] Saved diagnostics -> {diag_path}", flush=True)

            history_rows.append(
                {
                    "epoch": int(ep + 1),
                    "unit_key": str(batch.get("unit_key", "NA")),
                    "train_total": float(total_val),
                    "train_psth": float(psth_val),
                    "train_psth_weighted": float(psth_weighted_val),
                    "train_psth_sample": float(epoch_train_vals["sample"]),
                    "train_psth_delay": float(epoch_train_vals["delay"]),
                    "train_psth_response": float(epoch_train_vals["response"]),
                    "train_delay_shape_loss": float(delay_shape_val),
                    "train_delay_shape_weighted": float(weighted_delay_shape_val),
                    "train_type": float(reg_val),
                    "train_J": float(J_val),
                    "lambda_J_eff": 0.0,
                    "J_frob": float(J_frob),
                    "J_maxabs": float(J_maxabs),
                    "eval_total": float(eval_total),
                    "eval_psth": float(eval_psth),
                    "eval_psth_sample": float(eval_psth_sample),
                    "eval_psth_delay": float(eval_psth_delay),
                    "eval_psth_response": float(eval_psth_response),
                    "eval_type": 0.0,
                    "eval_J": float(eval_J),
                }
            )

            if do_eval:
                _flush_history_safe()
            if best_state is not None and int(save_best_every) > 0 and ((ep + 1) % int(save_best_every) == 0):
                _atomic_torch_save(best_state, best_path)
            if int(save_latest_every) > 0 and ((ep + 1) % int(save_latest_every) == 0):
                _atomic_torch_save(net.state_dict(), latest_path)
    except Exception:
        _flush_history_safe()
        raise

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        best_ep = int(max_epochs)
        best_score = float(best_total)

    _atomic_torch_save(best_state, best_path)
    _atomic_torch_save(best_state, model_path)
    _flush_history_safe()

    final_diag_path = save_blockrank_diagnostics(
        model=net,
        out_dir=diag_train_dir,
        stem="blockrank_diag_final",
        extra={
            "epoch": int(max_epochs),
            "best_epoch": int(best_ep),
            "best_metric": str(best_metric),
            "best_score": float(best_score),
            "type_names": list(block_info_full["type_names"]),
            "full_type_counts": dict(block_info_full["full_type_counts"]),
        },
    )

    meta_out = {
        "animal": str(animal),
        "registry_csv": str(registry_csv),
        "selected_unit_keys": list(selected_unit_keys),
        "global_idx_reindexed": True,
        "n_exc_virtual": int(n_exc_virtual),
        "n_obs_total": int(shared["N_obs_total"]),
        "n_total": int(n_total),
        "D_in": int(D_in),
        "dt": float(dt),
        "tau": float(tau),
        "substeps": int(substeps),
        "nonlinearity": str(nonlinearity),
        "model_type": str(model_type),
        "psth_target_source": str(psth_target_source),
        "block_rank": int(block_rank),
        "dale_strict": bool(dale_strict),
        "factor_nonlinearity": str(factor_nonlinearity),
        "normalize_uv": bool(normalize_uv),
        "A_nonlinearity": str(A_nonlinearity),
        "A_l2": float(A_l2),
        "uv_l2": float(uv_l2),
        "celltype_mode": str(celltype_mode),
        "block_celltype_min_count": int(block_celltype_min_count),
        "block_registry_label_cols": None if block_registry_label_cols is None else [str(x) for x in block_registry_label_cols],
        "other_inh_label": str(other_inh_label),
        "type_names": list(block_info_full["type_names"]),
        "type_signs": list(block_info_full["type_signs"]),
        "observed_type_counts": dict(block_info_obs["observed_type_counts"]),
        "full_type_counts": dict(block_info_full["full_type_counts"]),
        "registry_label_col_used": block_info_obs["registry_label_col_used"],
        "reliable_inh_labels": list(block_info_obs["reliable_inh_labels"]),
        "merged_inh_counts": dict(block_info_obs["merged_inh_counts"]),
        "block_list": list(block_info_full["block_list"]),
        "presyn_count_by_type": dict(block_info_full["presyn_count_by_type"]),
        "n_trainable_params": int(n_trainable_params),
        "n_recurrent_trainable_params": int(n_recurrent_trainable_params),
        "n_dense_equivalent_params": int(n_dense_equivalent),
        "psth_bin_ms": float(psth_bin_ms),
        "psth_smoothing": DEFAULT_PSTH_SMOOTHING,
        "psth_smoothing_kernel_frames": int(psth_kernel_frames),
        "viz_smooth_bin_ms": float(viz_smooth_bin_ms),
        "viz_smooth_kernel_frames": int(viz_kernel_frames),
        "sample_ignore_ms": float(sample_ignore_ms),
        "resp_sec": float(resp_sec),
        "loss_mode": str(loss_mode_norm),
        "loss_epoch_weights": dict(loss_epoch_weights_norm),
        "delay_shape_loss_type": str(delay_shape_loss_type),
        "delay_shape_lambda": float(delay_shape_lambda),
        "delay_shape_eps": float(delay_shape_eps),
        "delay_shape_min_scale": float(delay_shape_min_scale),
        "delay_shape_group_mode": str(delay_shape_group_mode),
        "diag_every": int(diag_every),
        "diag_subdir_train": str(diag_subdir_train),
        "diag_subdir_eval": str(diag_subdir_eval),
        "diag_train_dir": str(diag_train_dir),
        "diag_eval_dir": str(diag_eval_dir),
        "best_total": float(best_total),
        "best_train_step_epoch": int(best_train_ep),
        "best_epoch": int(best_ep),
        "best_metric": str(best_metric),
        "best_score": float(best_score),
        "best_eval_stats": best_eval_stats,
        "best_ckpt_path": str(best_path),
        "legacy_model_alias": str(model_path),
        "train_config_path": str(params_path),
        "train_history_csv": os.path.join(out_dir, "train_history.csv"),
        "loss_curves_png": os.path.join(out_dir, "loss_curves.png"),
        "final_diag_path": str(final_diag_path),
    }
    with open(meta_path, "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"[OK] Saved best model -> {best_path}", flush=True)
    print(f"[OK] Saved legacy alias -> {model_path}", flush=True)
    print(f"[OK] Saved meta -> {meta_path}", flush=True)
    print(f"[OK] Saved train config -> {params_path}", flush=True)
    print(f"[OK] Saved training history -> {os.path.join(out_dir, 'train_history.csv')}", flush=True)
    print(f"[OK] Saved final diagnostics -> {final_diag_path}", flush=True)

    return {
        "net": net,
        "units": units,
        "shared": shared,
        "meta_path": meta_path,
        "best_path": best_path,
        "model_path": model_path,
        "block_info_observed": block_info_obs,
        "block_info_full": block_info_full,
    }
