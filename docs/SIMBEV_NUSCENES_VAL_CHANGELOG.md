# SimBEV `nuscenes_val` Adaptation Notes

This document records the recent code changes made to run/evaluate UniTR on custom SimBEV-generated `nuscenes_val` data and to debug related runtime errors.

## Scope

- Dataset path adaptation for SimBEV json infos that contain machine-specific absolute paths.
- Checkpoint loading compatibility with newer PyTorch (`torch.load` behavior change).
- Data-to-GPU conversion robustness for object-typed numpy entries.
- UniTR image-to-lidar debug instrumentation.
- UniTR SetAttention fix for `-1` set indices and incomplete index coverage.

## Code Changes

### 1) SimBEV path normalization and pickle compatibility

File: `pcdet/datasets/simbev/simbev_dataset.py`

- Added annotation path normalization after loading infos:
  - `normalize_info_paths(...)`
  - `resolve_data_path(...)`
- Supports mapping legacy path roots to current `DATA_PATH`.
- Added safe loader for object-array annotation files:
  - `safe_load_pickled_npy(...)`
  - Fallback context manager `numpy_pickle_compat()` for `numpy._core` pickle alias issues.
- `load_gt_bboxes(...)` now uses `safe_load_pickled_npy(...)` instead of direct `np.load(..., allow_pickle=True)`.

### 2) Checkpoint loading compatibility (PyTorch >= 2.6)

File: `pcdet/models/detectors/detector3d_template.py`

- Added helper `_torch_load_checkpoint(...)`.
- Uses `torch.load(..., weights_only=False)` when supported, with fallback for older versions.
- Replaced direct `torch.load(...)` in model/pretrain/optimizer checkpoint load paths with the helper.

### 3) Safer GPU batch conversion

File: `pcdet/models/__init__.py`

- Added `ori_imgs` to keys skipped by `load_data_to_gpu(...)`.
- Added explicit error for unexpected `np.object_` arrays:
  - raises `TypeError` with key name and shape for faster diagnosis.

### 4) UniTR image2lidar debug instrumentation

File: `pcdet/models/mm_backbone/unitr.py`

- Added optional debug output in `_image2lidar_preprocess(...)`.
- Enabled by environment variable:
  - `UNITR_DEBUG_IMAGE2LIDAR=1`
- Debug prints include:
  - total token counts (`multi_feat`, `voxel_num`, `patch_num`)
  - per-shift window density stats
  - per shift/set index coverage, `-1` counts, mask counts, min/max index

### 5) UniTR SetAttention robustness for `-1` indices

File: `pcdet/models/mm_backbone/unitr.py`

- In `SetAttention.forward(...)`:
  - `voxel_inds` are clamped for safe gather.
  - `invalid_inds_mask = voxel_inds < 0` merged into attention key padding mask.
  - Attention output is reconstructed into a full-length tensor (`src2_full`) with length equal to `src`.
  - Uncovered tokens remain zero update (identity via residual), preventing shape mismatch in `src + src2`.

## Local Config/CLI Tweaks Also Present

The following local modifications are currently in the workspace and may be environment-specific:

- `tools/cfgs/dataset_configs/simbev_dataset.yaml`
  - `DATA_PATH` changed to `../data/simbev/original`
- `tools/cfgs/simbev_models/unitr.yaml`
  - `OPTIMIZATION.BATCH_SIZE_PER_GPU: 6 -> 4`
- `tools/cfgs/simbev_models/unitr+lss.yaml`
  - `OPTIMIZATION.BATCH_SIZE_PER_GPU: 6 -> 4`
- `tools/test.py`
  - argument name changed from `--local_rank` to `--local-rank`

## Useful Run Example

```bash
cd tools
UNITR_DEBUG_IMAGE2LIDAR=1 bash scripts/dist_test.sh 1 \
  --cfg_file cfgs/simbev_models/unitr.yaml \
  --ckpt ../checkpoints/unitr-det.pth \
  --set DATA_CONFIG.DATA_PATH ../data/simbev/setup/nuscenes_val
```

## Notes

- `-1` in set indices comes from set-padding in DSVT-style set construction; it is expected as padding, but previously could cause incomplete reconstruction in SetAttention.
- Worker `BrokenPipe` / `Aborted` errors are usually secondary after the main exception in model forward.
