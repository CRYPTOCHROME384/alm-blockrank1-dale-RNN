# ALM BlockRank1 Dale RNN

This repository contains the current development code for ALM current-based RNN experiments centered on a cell-type block rank-1 Dale-constrained recurrent model.

## Main Contents

- `current_rnn/`: training, evaluation, model, and data pipeline code
- `tests/`: smoke tests and delay-dynamics unit tests
- `tools/`: helper scripts for launchers and dataset cache export
- `README_BLOCKRANK1_DALE.md`: project-specific notes and workflow details

## Notes

- Training results and logs are intentionally excluded from version control.
- The active model implementation lives in `current_rnn/models/celltype_block_rank1_dale.py`.
- The main training entrypoint is `current_rnn/main_train_alm_current_blockrank1_dale.py`.
