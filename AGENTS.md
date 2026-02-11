# Repository Guidelines

## Project Structure & Module Organization
- `pcdet/`: core library code (datasets, models, CUDA ops, utils, config helpers).
- `tools/`: runnable entry points such as `train.py`, `test.py`, eval utilities, and launch scripts in `tools/scripts/`.
- `tools/cfgs/`: experiment configs grouped by dataset/model family (for this fork, use `simbev2nuscenes_models/` for SimBEV-to-nuScenes workflows).
- `data/`: dataset split files and dataset-local metadata.
- `docs/` and `assets/`: setup docs and figures.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies.
- `python setup.py develop`: install `pcdet` in editable mode and compile CUDA extensions.
- `python tools/train.py --cfg_file tools/cfgs/simbev2nuscenes_models/unitr.yaml --launcher none`: single-process training.
- `bash tools/scripts/dist_train.sh 4 --cfg_file tools/cfgs/simbev2nuscenes_models/unitr.yaml`: distributed training on 4 GPUs.
- `python tools/test.py --cfg_file tools/cfgs/simbev2nuscenes_models/unitr.yaml --ckpt /path/to/checkpoint.pth --launcher none`: evaluate one checkpoint.
- `bash tools/scripts/dist_test.sh 4 --cfg_file ... --ckpt ...`: multi-GPU evaluation.

## Coding Style & Naming Conventions
- Python uses 4-space indentation and follows PEP 8 style; keep functions small and config-driven.
- Use `snake_case` for functions/variables/files, `PascalCase` for classes, and `UPPER_CASE` for constants.
- YAML config names should be descriptive and task-specific (examples: `unitr.yaml`, `unitr_map+lss.yaml`).
- Keep dataset/model-specific logic inside the matching submodule (`pcdet/datasets/*`, `pcdet/models/*`).

## Testing Guidelines
- There is no standalone unit-test suite in this fork; validation is done through `tools/test.py` and benchmark metrics.
- Before opening a PR, run at least one train/eval smoke check with your target config and verify logs under `output/.../eval/`.
- When changing dataset processing or metrics, include before/after metric snippets in the PR.

## Commit & Pull Request Guidelines
- Recent history favors short, imperative commit titles (e.g., `Added confusion matrix for BEV segmentation`, `Fixed typo.`).
- Keep commits focused to one logical change and mention affected area (`datasets`, `eval`, `docker`, etc.).
- PRs should include: purpose, key changes, exact commands used, config/checkpoint paths, and resulting metrics.
- Link related issues and attach screenshots only for visualization/UI output changes.

## Security & Configuration Tips
- Do not commit datasets, checkpoints, or secrets; keep machine-specific paths out of tracked YAMLs.
- Prefer passing overrides via CLI (`--set KEY VALUE`) instead of hardcoding environment-specific values.
