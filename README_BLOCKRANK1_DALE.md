# ALM BlockRank1 Dale Current-RNN

This workspace adds a new deterministic ALM current-RNN variant without touching the stable baseline in `/home/jingyi.xu/code_rnn_1230_repro`.

Project root:
- `/home/jingyi.xu/code_rnn_blockrank1_dale`

Stable pieces intentionally reused:
- Phase-2 `trials_mean` / PSTH alignment pipeline
- mixed-EI registry loading
- explicit `cell_sign` / `cell_subclass` reading
- deterministic current-RNN train/eval structure

## Model

Dynamics are unchanged from the current-based RNN:

```text
tau * dh_i/dt =
    -h_i
    + recurrent_input_i
    + input_i(t)
    + b_i
```

Only the recurrent matrix parameterization is replaced.

For postsynaptic type `a` and presynaptic type `b`:

```text
u_pos^{a<-b}_i = softplus(u_raw^{a<-b}_i)
v_pos^{a<-b}_j = softplus(v_raw^{a<-b}_j)

u_norm^{a<-b}_i = u_pos^{a<-b}_i / (mean_i(u_pos^{a<-b}_i) + eps)
v_norm^{a<-b}_j = v_pos^{a<-b}_j / (mean_j(v_pos^{a<-b}_j) + eps)

A_pos^{a<-b} = softplus(A_raw^{a<-b})

J^{a<-b}_{ij} =
    s_b * A_pos^{a<-b} * u_norm^{a<-b}_i * v_norm^{a<-b}_j / N_b
```

Where:
- `s_b = +1` for `E_broad`
- `s_b = -1` for inhibitory subclasses
- `N_b` is the presynaptic cell count for type `b`

Forward pass is efficient and does not build dense `J` during training:

```text
kappa^{a<-b} = sum_{j in C_b} v_norm^{a<-b}_j * phi(h_j) / N_b
input_i_from_b = s_b * A_pos^{a<-b} * u_norm^{a<-b}_i * kappa^{a<-b}
```

## Cell Types

`celltype_mode="broadE_inh_subclass"` is used in v1:
- all excitatory cells merge into `E_broad`
- inhibitory cells keep reliable subclasses from the registry
- rare or missing inhibitory labels are merged into `I_other`

For the current mixed-EI registry used in smoke runs, the observed types were:
- `E_broad`, `Lamp5`, `Meis2`, `Pvalb`, `Sncg`, `Sst`, `Vip`

## Main Files

- Model: [celltype_block_rank1_dale.py](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/models/celltype_block_rank1_dale.py)
- Train entry: [main_train_alm_current_blockrank1_dale.py](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/main_train_alm_current_blockrank1_dale.py)
- Train loop: [training_blockrank1_dale.py](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/training_blockrank1_dale.py)
- Eval entry: [eval_blockrank1_dale.py](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/eval_blockrank1_dale.py)
- Block/type helpers: [blockrank_dale_utils.py](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/blockrank_dale_utils.py)
- Main config: [parameters_ei_mixed_blockrank1_dale.json](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale.json)
- Delay `1:2:1` config: [parameters_ei_mixed_blockrank1_dale_delay121.json](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale_delay121.json)
- Delay `1:3:1` config: [parameters_ei_mixed_blockrank1_dale_delay131.json](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale_delay131.json)
- Smoke config: [parameters_ei_mixed_blockrank1_dale_smoke.json](/home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale_smoke.json)
- Single launcher: [launch_train_blockrank1_dale.sh](/home/jingyi.xu/code_rnn_blockrank1_dale/tools/launch_train_blockrank1_dale.sh)
- Pair launcher: [launch_train_blockrank1_dale_pair.sh](/home/jingyi.xu/code_rnn_blockrank1_dale/tools/launch_train_blockrank1_dale_pair.sh)
- Trials cache helper: [export_registry_trials_npz.py](/home/jingyi.xu/code_rnn_blockrank1_dale/tools/export_registry_trials_npz.py)
- Unit test: [test_celltype_block_rank1_dale.py](/home/jingyi.xu/code_rnn_blockrank1_dale/tests/test_celltype_block_rank1_dale.py)

## Important Config Keys

- `model_type = "celltype_block_rank1_dale"`
- `psth_target_source = "trials_mean"`
- `use_trials = true`
- `strict_trials_align = true`
- `trials_root = "/home/jingyi.xu/ALM/results/trials_registry_cache"`
- `block_rank = 1`
- `dale_strict = true`
- `factor_nonlinearity = "softplus"`
- `normalize_uv = true`
- `A_nonlinearity = "softplus"`
- `A_l2`, `uv_l2` optional and weak by default
- `loss_mode = "epoch_weighted_mean"`
- `loss_epoch_weights = {"sample": 1.0, "delay": 2.0 or 3.0, "response": 1.0}`
- `diag_every`, `diag_subdir_train`, `diag_subdir_eval`
- `viz_smooth_bin_ms = 400.0`
- `use_x0_noise = false`
- `use_process_noise = false`
- `lambda_var = 0.0`
- `lambda_celltype = 0.0`

## Trials Cache

The stable Phase-2 loader expects `trials_<session>.<plane>.npz`. For the mixed-EI registry, an independent cache was generated here:

- `/home/jingyi.xu/ALM/results/trials_registry_cache`

Regenerate it with:

```bash
python /home/jingyi.xu/code_rnn_blockrank1_dale/tools/export_registry_trials_npz.py \
  --registry-dir /home/jingyi.xu/ALM/results/registry/ei_observed_kd53_kd91_kd95_e800_i200 \
  --animal ei_observed_kd53_kd91_kd95_e800_i200 \
  --out-root /home/jingyi.xu/ALM/results/trials_registry_cache \
  --force
```

## Train

Full config:

```bash
python /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/main_train_alm_current_blockrank1_dale.py \
  --config /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale.json
```

Parallel delay-weight comparison:

```bash
/home/jingyi.xu/code_rnn_blockrank1_dale/tools/launch_train_blockrank1_dale_pair.sh
```

Single delayed-weight run on a chosen GPU:

```bash
/home/jingyi.xu/code_rnn_blockrank1_dale/tools/launch_train_blockrank1_dale.sh \
  /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale_delay121.json \
  0 \
  delay121_manual \
  delay121_manual
```

Smoke config:

```bash
python /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/main_train_alm_current_blockrank1_dale.py \
  --config /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/parameters_ei_mixed_blockrank1_dale_smoke.json
```

If `out_dir` is null, a fresh directory is created under:

- `/home/jingyi.xu/code_rnn_blockrank1_dale/results_current/blockrank1_dale_<timestamp>/`

## Eval

```bash
python /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/eval_blockrank1_dale.py \
  --model /path/to/rnn_current_*.best.pt \
  --out_dir /path/to/eval_best
```

## Smoke Test

Unit test:

```bash
python -m unittest discover \
  -s /home/jingyi.xu/code_rnn_blockrank1_dale/tests \
  -p 'test_celltype_block_rank1_dale.py'
```

Syntax check:

```bash
python -m py_compile \
  /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/models/celltype_block_rank1_dale.py \
  /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/training_blockrank1_dale.py \
  /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/eval_blockrank1_dale.py \
  /home/jingyi.xu/code_rnn_blockrank1_dale/current_rnn/main_train_alm_current_blockrank1_dale.py \
  /home/jingyi.xu/code_rnn_blockrank1_dale/tools/export_registry_trials_npz.py
```

## Outputs

Training directory contains:
- checkpoints: `*.best.pt`, `*.latest.pt`, `*.pt`
- runtime config: `parameters.launch.json`, `*.train_config.json`, `*.meta.json`
- training history: `train_history.csv`, `loss_curves.png`
- diagnostics: `diagnostics/train/blockrank_diag_ep*.json`, `blockrank_diag_best.json`, `blockrank_diag_final.json`

Eval directory contains:
- `eval_summary_*.json`
- `eval_per_unit_*.csv`
- `eval_per_celltype_*.csv`
- `diagnostics/eval/blockrank_eval_diag_*.json`
- `psth_eval_all_best_mosaic_*_R0.png`
- `psth_eval_all_best_mosaic_*_R0_vizsmooth.png`

Diagnostics include:
- `dale_violation_count`
- `numerical_block_ranks`
- `block_parameter_summary`
  - `A_pos`
  - `u_norm_mean/std/min/max`
  - `v_norm_mean/std/min/max`
- optional reconstructed dense-`J` summary

## Smoke Run Used Here

Smoke train/eval completed in:
- `/home/jingyi.xu/code_rnn_blockrank1_dale/results_current/blockrank1_dale_20260501_195446`

Eval outputs are in:
- `/home/jingyi.xu/code_rnn_blockrank1_dale/results_current/blockrank1_dale_20260501_195446/eval_best`
