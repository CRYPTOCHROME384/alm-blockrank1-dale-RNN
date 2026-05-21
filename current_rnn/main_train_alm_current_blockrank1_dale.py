import argparse
import json
import os
import time
from copy import deepcopy
'''
nohup python -u /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/main_train_alm_current_blockrank1_dale.py \
  --config /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale.json \
  --cuda_device 0 \
  --device cuda \
  --require_cuda true \
  > /home/jingyi.xu/code_rnn_blockrank1_dale/logs/train_ei_mixed_s6_20260502.log 2>&1 &
  '''
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a single JSON config file.")
    ap.add_argument("--out_dir", default=None, help="Optional override for config['out_dir'].")
    ap.add_argument("--out_dir_suffix", default=None, help="Optional suffix appended to the default results_current directory name.")
    ap.add_argument("--registry_dir", default=None, help="Optional override for config['registry_dir'].")
    ap.add_argument("--animal", default=None, help="Optional override for config['animal'].")
    ap.add_argument("--device", default=None, help="Optional override for config['device'] (auto/cpu/cuda/cuda:0).")
    ap.add_argument("--cuda_device", default=None, help="Optional override for CUDA_VISIBLE_DEVICES, e.g. 0 or 1.")
    ap.add_argument("--require_cuda", default=None, help="Optional override for config['require_cuda'] (true/false).")
    return ap.parse_args()


def _load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _as_cond_filter(x):
    if x is None:
        return None
    if isinstance(x, list):
        y = [str(v).strip() for v in x if str(v).strip() != ""]
        return y if len(y) > 0 else None
    s = str(x).strip()
    if s == "":
        return None
    return [t.strip() for t in s.split(",") if t.strip()]


def _default_out_dir(suffix: str = "") -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suffix_norm = str(suffix or "").strip().strip("_")
    name = f"blockrank1_dale_{suffix_norm}_{stamp}" if suffix_norm != "" else f"blockrank1_dale_{stamp}"
    return os.path.join(PROJECT_ROOT, "results_current", name)


def _parse_bool_arg(x):
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean override from {x!r}")


def main():
    args = _parse_args()
    cfg = _load_json(args.config)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object/dict.")

    cfg = deepcopy(cfg)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.out_dir_suffix is not None:
        cfg["out_dir_suffix"] = args.out_dir_suffix
    if args.registry_dir is not None:
        cfg["registry_dir"] = args.registry_dir
    if args.animal is not None:
        cfg["animal"] = args.animal
    if args.device is not None:
        cfg["device"] = args.device
    if args.cuda_device is not None:
        cfg["cuda_device"] = args.cuda_device
    req_cuda_override = _parse_bool_arg(args.require_cuda)
    if req_cuda_override is not None:
        cfg["require_cuda"] = bool(req_cuda_override)

    required = ["registry_dir", "animal"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    cfg.setdefault("out_dir", None)
    cfg.setdefault("out_dir_suffix", None)
    if cfg["out_dir"] in (None, ""):
        cfg["out_dir"] = _default_out_dir(str(cfg.get("out_dir_suffix", "") or ""))

    cfg.setdefault("device", "auto")
    cfg.setdefault("cuda_device", None)
    cfg.setdefault("require_cuda", False)
    cfg.setdefault("enable_tf32", True)
    cfg.setdefault("cudnn_benchmark", True)
    cfg.setdefault("matmul_precision", "high")

    cuda_device = cfg.get("cuda_device", None)
    if cuda_device not in (None, ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
        print(f"[INFO] Set CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)

    cfg.setdefault("max_epochs", 30000)
    cfg.setdefault("lr", 5e-5)
    cfg.setdefault("weight_decay", 0.0)
    cfg.setdefault("seed", 42)
    cfg.setdefault("grad_clip", 1.0)
    cfg.setdefault("print_every", 50)
    cfg.setdefault("dt", 0.03436)
    cfg.setdefault("tau", 0.01)
    cfg.setdefault("substeps", 6)
    cfg.setdefault("nonlinearity", "tanh")
    cfg.setdefault("n_exc_virtual", 0)
    cfg.setdefault("unit_sampling", "cycle")
    cfg.setdefault("max_sessions", 0)
    cfg.setdefault("cond_filter", ["left_correct", "right_correct"])
    cfg.setdefault("max_time", None)
    cfg.setdefault("psth_bin_ms", 200.0)
    cfg.setdefault("sample_ignore_ms", 50.0)
    cfg.setdefault("resp_sec", 2.0)
    cfg.setdefault("save_best_every", 200)
    cfg.setdefault("save_latest_every", 0)
    cfg.setdefault("eval_every", 200)
    cfg.setdefault("best_metric", "eval_psth")
    cfg.setdefault("loss_mode", "epoch_weighted_mean")
    cfg.setdefault("loss_epoch_weights", {"sample": 1.0, "delay": 2.0, "response": 1.0})
    cfg.setdefault("diag_every", 1000)
    cfg.setdefault("diag_subdir_train", "diagnostics/train")
    cfg.setdefault("diag_subdir_eval", "diagnostics/eval")
    cfg.setdefault("viz_smooth_bin_ms", 400.0)
    cfg.setdefault("delay_shape_loss_type", "none")
    cfg.setdefault("delay_shape_lambda", 0.0)
    cfg.setdefault("delay_shape_eps", 1e-8)
    cfg.setdefault("delay_shape_min_scale", 1e-6)
    cfg.setdefault("delay_shape_group_mode", "all")

    cfg.setdefault("model_type", "celltype_block_rank1_dale")
    cfg.setdefault("psth_target_source", "trials_mean")
    cfg.setdefault("use_trials", str(cfg["psth_target_source"]) == "trials_mean")
    cfg.setdefault("strict_trials_align", True)
    cfg.setdefault("trials_root", None)
    cfg.setdefault("trials_path_mode", "auto_from_stage1")
    cfg.setdefault("trial_keys", None)
    cfg.setdefault("debug_trials_align", False)

    cfg.setdefault("use_x0_noise", False)
    cfg.setdefault("use_process_noise", False)
    cfg.setdefault("lambda_var", 0.0)
    cfg.setdefault("lambda_celltype", 0.0)

    cfg.setdefault("block_rank", 1)
    cfg.setdefault("dale_strict", True)
    cfg.setdefault("factor_nonlinearity", "softplus")
    cfg.setdefault("normalize_uv", True)
    cfg.setdefault("A_nonlinearity", "softplus")
    cfg.setdefault("A_l2", 0.0)
    cfg.setdefault("uv_l2", 0.0)
    cfg.setdefault("celltype_mode", "broadE_inh_subclass")
    cfg.setdefault("block_celltype_min_count", 10)
    cfg.setdefault("block_registry_label_cols", ["cell_subclass", "cell_cluster", "cell_type", "celltype"])
    cfg.setdefault("other_inh_label", "I_other")
    cfg.setdefault("init_A", 0.10)
    cfg.setdefault("init_factor_scale", 0.02)
    cfg.setdefault("eps", 1e-8)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    launch_cfg_path = os.path.join(cfg["out_dir"], "parameters.launch.json")
    with open(launch_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Saved launch config -> {launch_cfg_path}", flush=True)

    from training_blockrank1_dale import train_current_alm_global_blockrank1_dale

    train_current_alm_global_blockrank1_dale(
        registry_dir=str(cfg["registry_dir"]),
        animal=str(cfg["animal"]),
        out_dir=str(cfg["out_dir"]),
        device=str(cfg["device"]),
        require_cuda=bool(cfg["require_cuda"]),
        enable_tf32=bool(cfg["enable_tf32"]),
        cudnn_benchmark=bool(cfg["cudnn_benchmark"]),
        matmul_precision=str(cfg["matmul_precision"]),
        max_epochs=int(cfg["max_epochs"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        seed=int(cfg["seed"]),
        grad_clip=(None if cfg["grad_clip"] is None else float(cfg["grad_clip"])),
        print_every=int(cfg["print_every"]),
        dt=float(cfg["dt"]),
        tau=float(cfg["tau"]),
        substeps=int(cfg["substeps"]),
        nonlinearity=str(cfg["nonlinearity"]),
        n_exc_virtual=int(cfg["n_exc_virtual"]),
        unit_sampling=str(cfg["unit_sampling"]),
        max_sessions=(None if int(cfg["max_sessions"]) <= 0 else int(cfg["max_sessions"])),
        cond_filter=_as_cond_filter(cfg.get("cond_filter", None)),
        max_time=cfg.get("max_time", None),
        psth_bin_ms=float(cfg["psth_bin_ms"]),
        sample_ignore_ms=float(cfg["sample_ignore_ms"]),
        resp_sec=float(cfg["resp_sec"]),
        save_best_every=int(cfg["save_best_every"]),
        save_latest_every=int(cfg["save_latest_every"]),
        eval_every=int(cfg["eval_every"]),
        best_metric=str(cfg["best_metric"]),
        loss_mode=str(cfg["loss_mode"]),
        loss_epoch_weights=cfg.get("loss_epoch_weights", None),
        use_trials=bool(cfg["use_trials"]),
        strict_trials_align=bool(cfg["strict_trials_align"]),
        trials_root=cfg.get("trials_root", None),
        trials_path_mode=str(cfg["trials_path_mode"]),
        trial_keys=cfg.get("trial_keys", None),
        debug_trials_align=bool(cfg["debug_trials_align"]),
        model_type=str(cfg["model_type"]),
        psth_target_source=str(cfg["psth_target_source"]),
        block_rank=int(cfg["block_rank"]),
        dale_strict=bool(cfg["dale_strict"]),
        factor_nonlinearity=str(cfg["factor_nonlinearity"]),
        normalize_uv=bool(cfg["normalize_uv"]),
        A_nonlinearity=str(cfg["A_nonlinearity"]),
        A_l2=float(cfg["A_l2"]),
        uv_l2=float(cfg["uv_l2"]),
        celltype_mode=str(cfg["celltype_mode"]),
        block_celltype_min_count=int(cfg["block_celltype_min_count"]),
        block_registry_label_cols=cfg.get("block_registry_label_cols", None),
        other_inh_label=str(cfg["other_inh_label"]),
        init_A=float(cfg["init_A"]),
        init_factor_scale=float(cfg["init_factor_scale"]),
        eps=float(cfg["eps"]),
        use_x0_noise=bool(cfg["use_x0_noise"]),
        use_process_noise=bool(cfg["use_process_noise"]),
        lambda_var=float(cfg["lambda_var"]),
        lambda_celltype=float(cfg["lambda_celltype"]),
        diag_every=int(cfg["diag_every"]),
        diag_subdir_train=str(cfg["diag_subdir_train"]),
        diag_subdir_eval=str(cfg["diag_subdir_eval"]),
        viz_smooth_bin_ms=float(cfg["viz_smooth_bin_ms"]),
        delay_shape_loss_type=str(cfg["delay_shape_loss_type"]),
        delay_shape_lambda=float(cfg["delay_shape_lambda"]),
        delay_shape_eps=float(cfg["delay_shape_eps"]),
        delay_shape_min_scale=float(cfg["delay_shape_min_scale"]),
        delay_shape_group_mode=str(cfg["delay_shape_group_mode"]),
        run_config=cfg,
    )


if __name__ == "__main__":
    main()
