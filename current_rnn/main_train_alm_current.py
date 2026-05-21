# current_rnn/main_train_alm_current.py
"""
Single-config entrypoint (registry-global condition-average PSTH training + optional celltype loss A).

Usage:
nohup python -u main_train_alm_current.py --config /home/jingyi.xu/code_rnn_1230_repro/results_current/run_configs/2026-04-22/ei_mixed_s8_150k.json > /home/jingyi.xu/code_rnn_1230_repro/logs/train_ei_mixed_s8_20260422.log 2>&1 &

Notes
-----
- All training parameters live in one JSON file (no split CLI hyperparameters).
- A resolved copy of the config is saved into out_dir together with checkpoints.
- This entrypoint intentionally targets the registry-global condition-average PSTH path
"""

import os
import json
import argparse
from copy import deepcopy

from training_current import train_current_alm_global


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a single JSON config file.")
    # optional lightweight overrides for convenience (keep small and explicit)
    ap.add_argument("--out_dir", default=None, help="Optional override for config['out_dir']")
    ap.add_argument("--registry_dir", default=None, help="Optional override for config['registry_dir']")
    ap.add_argument("--animal", default=None, help="Optional override for config['animal']")
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


def main():
    args = _parse_args()
    cfg = _load_json(args.config)
    if not isinstance(cfg, dict):
        raise ValueError("Config JSON must be an object/dict.")

    cfg = deepcopy(cfg)
    if args.out_dir is not None:
        cfg["out_dir"] = args.out_dir
    if args.registry_dir is not None:
        cfg["registry_dir"] = args.registry_dir
    if args.animal is not None:
        cfg["animal"] = args.animal

    required = ["registry_dir", "animal", "out_dir"]
    missing = [k for k in required if k not in cfg or cfg[k] in (None, "")]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    os.makedirs(cfg["out_dir"], exist_ok=True)

    # Phase2 trials options (defaults preserve baseline behavior when use_trials=false).
    cfg.setdefault("use_trials", False)
    cfg.setdefault("strict_trials_align", True)
    cfg.setdefault("trials_root", None)
    cfg.setdefault("trials_path_mode", "auto_from_stage1")
    cfg.setdefault("trial_keys", None)
    cfg.setdefault("debug_trials_align", False)
    # Phase3A variability modeling (defaults preserve Phase2 behavior).
    cfg.setdefault("phase3_enable_var_loss", False)
    cfg.setdefault("noise_mode", "none")  # "none" | "x0_gaussian"
    cfg.setdefault("celltype_loss_on", "mean_only")
    cfg.setdefault("sigma_x0", 0.0)
    cfg.setdefault("n_sim_trials_train", 16)
    cfg.setdefault("n_sim_trials_eval", 32)
    cfg.setdefault("lambda_var", 0.0)
    cfg.setdefault("lambda_var_warmup_epochs", 100)
    cfg.setdefault("lambda_var_ramp_epochs", 200)
    cfg.setdefault("var_loss_space", "logvar")  # "var" | "logvar"
    cfg.setdefault("var_loss_eps", 1e-6)
    cfg.setdefault("var_unbiased", False)
    cfg.setdefault("min_trials_for_var_real", 3)
    cfg.setdefault("var_loss_weighting", "sqrt_nminus1")  # "none" | "sqrt_nminus1" | "nminus1"
    cfg.setdefault("eval_seed_base", 12345)
    cfg.setdefault("fixed_eval_seed_bank", True)
    cfg.setdefault("debug_var_loss", False)
    cfg.setdefault("debug_noise", False)
    cfg.setdefault("lambda_J", 0.0)
    cfg.setdefault("lambda_J_warmup_epochs", 0)
    cfg.setdefault("lambda_J_ramp_epochs", 0)
    cfg.setdefault("recurrent_mode", "full")
    cfg.setdefault("recurrent_rank", 0)
    cfg.setdefault("random_bg_scale", 0.0)

    # Save the exact config used to launch this run (before training starts).
    launch_cfg_path = os.path.join(cfg["out_dir"], "parameters.launch.json")
    with open(launch_cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[INFO] Saved launch config -> {launch_cfg_path}", flush=True)

    train_current_alm_global(
        registry_dir=str(cfg["registry_dir"]),
        animal=str(cfg["animal"]),
        out_dir=str(cfg["out_dir"]),
        max_epochs=int(cfg.get("max_epochs", 200000)),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        seed=int(cfg.get("seed", 42)),
        noise_std=float(cfg.get("noise_std", 0.0)),
        grad_clip=(None if cfg.get("grad_clip", 1.0) is None else float(cfg.get("grad_clip", 1.0))),
        print_every=int(cfg.get("print_every", 50)),
        dt=float(cfg.get("dt", 0.03436)),
        tau=float(cfg.get("tau", 0.01)),
        substeps=int(cfg.get("substeps", 4)),
        nonlinearity=str(cfg.get("nonlinearity", "tanh")),
        dale=bool(cfg.get("dale", True)),
        n_exc_virtual=int(cfg.get("n_exc_virtual", 800)),
        recurrent_mode=str(cfg.get("recurrent_mode", "full")),
        recurrent_rank=int(cfg.get("recurrent_rank", 0)),
        random_bg_scale=float(cfg.get("random_bg_scale", 0.0)),
        unit_sampling=str(cfg.get("unit_sampling", "random")),
        max_sessions=(None if int(cfg.get("max_sessions", 0)) <= 0 else int(cfg.get("max_sessions", 0))),
        cond_filter=_as_cond_filter(cfg.get("cond_filter", None)),
        max_time=cfg.get("max_time", None),
        psth_bin_ms=float(cfg.get("psth_bin_ms", 200.0)),
        sample_ignore_ms=float(cfg.get("sample_ignore_ms", 50.0)),
        resp_sec=float(cfg.get("resp_sec", 2.0)),
        lambda_J=float(cfg.get("lambda_J", 0.0)),
        lambda_J_warmup_epochs=int(cfg.get("lambda_J_warmup_epochs", 0)),
        lambda_J_ramp_epochs=int(cfg.get("lambda_J_ramp_epochs", 0)),
        lambda_celltype=float(cfg.get("lambda_celltype", 0.0)),
        celltype_label_keys=list(cfg.get("celltype_label_keys", ["cell_types", "cell_subclasses"])),
        celltype_exclude=list(cfg.get("celltype_exclude", ["", "nan", "none", "unknown"])),
        celltype_min_count=int(cfg.get("celltype_min_count", 2)),
        save_best_every=int(cfg.get("save_best_every", 100)),
        save_latest_every=int(cfg.get("save_latest_every", 0)),
        use_trials=bool(cfg.get("use_trials", False)),
        strict_trials_align=bool(cfg.get("strict_trials_align", True)),
        trials_root=cfg.get("trials_root", None),
        trials_path_mode=str(cfg.get("trials_path_mode", "auto_from_stage1")),
        trial_keys=cfg.get("trial_keys", None),
        debug_trials_align=bool(cfg.get("debug_trials_align", False)),
        phase3_enable_var_loss=bool(cfg.get("phase3_enable_var_loss", False)),
        noise_mode=str(cfg.get("noise_mode", "none")),
        celltype_loss_on=str(cfg.get("celltype_loss_on", "mean_only")),
        sigma_x0=float(cfg.get("sigma_x0", 0.0)),
        n_sim_trials_train=int(cfg.get("n_sim_trials_train", 16)),
        n_sim_trials_eval=int(cfg.get("n_sim_trials_eval", 32)),
        lambda_var=float(cfg.get("lambda_var", 0.0)),
        lambda_var_warmup_epochs=int(cfg.get("lambda_var_warmup_epochs", 100)),
        lambda_var_ramp_epochs=int(cfg.get("lambda_var_ramp_epochs", 200)),
        var_loss_space=str(cfg.get("var_loss_space", "logvar")),
        var_loss_eps=float(cfg.get("var_loss_eps", 1e-6)),
        var_unbiased=bool(cfg.get("var_unbiased", False)),
        min_trials_for_var_real=int(cfg.get("min_trials_for_var_real", 3)),
        var_loss_weighting=str(cfg.get("var_loss_weighting", "sqrt_nminus1")),
        eval_seed_base=int(cfg.get("eval_seed_base", 12345)),
        fixed_eval_seed_bank=bool(cfg.get("fixed_eval_seed_bank", True)),
        debug_var_loss=bool(cfg.get("debug_var_loss", False)),
        debug_noise=bool(cfg.get("debug_noise", False)),
        run_config=cfg,
    )


if __name__ == "__main__":
    main()
