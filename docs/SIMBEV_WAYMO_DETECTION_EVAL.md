# SimBEV Waymo Detection Eval Support

This note documents the current codebase changes for **SimBEV Waymo** support focused on **detection-only evaluation**.

## Scope

- In scope: 3D detection eval (`tools/test.py`, `tools/scripts/dist_test.sh`)
- Out of scope: map segmentation, training stabilization, CUDA toolchain migration

## Implemented Changes

### 1) SimBEV dataset compatibility updates

File: `pcdet/datasets/simbev/simbev_dataset.py`

- Dynamic camera selection per-sample:
  - Uses `RGB-*` keys in infos intersected with metadata camera keys.
  - No fixed 6-camera assumption.
- Camera intrinsics by camera name:
  - Prefer `metadata.camera_intrinsics_by_name[camera]`.
  - Fallback to `metadata.camera_intrinsics`.
- LiDAR key compatibility:
  - Supports both `LIDAR` and `LIDAR_TOP`.
- Path normalization retained and extended:
  - `ground-truth` <-> `ground_truth`
  - `sweeps` <-> `samples`
  - absolute path prefix remap to current `DATA_PATH`.
- Optional multi-camera image padding:
  - Controlled by `CAMERA_CONFIG.IMAGE.PAD_TO_MAX_SHAPE`.
  - Side-camera/heterogeneous image sizes are padded to sample-local max shape.
  - Principal point (`cx/cy`) is corrected after padding.
  - `lidar2image` is recomputed after intrinsic updates.

### 2) Model-side robustness kept enabled

File: `pcdet/models/mm_backbone/unitr.py`

- SetAttention robust handling for invalid set indices (`-1`) and incomplete coverage.
- Preserved image2lidar debug switch:
  - `UNITR_DEBUG_IMAGE2LIDAR=1`

### 3) LSS/DepthLSS dynamic camera-view support

Files:

- `pcdet/models/view_transforms/lss.py`
- `pcdet/models/view_transforms/depth_lss.py`

Changes:

- Removed hardcoded 6-view reshape logic:
  - `view(int(BN/6), 6, C, H, W)`
- Replaced with dynamic view count from runtime batch:
  - `num_views = batch_dict['camera_imgs'].shape[1]`
  - `view(BN // num_views, num_views, C, H, W)`
- Added divisibility checks (`BN % num_views == 0`) with explicit error messages.
- Updated `depth_lss.py` batch-size restoration from `BN // 6` to `BN // num_views`.

### 4) Data-to-GPU safeguard

File: `pcdet/models/__init__.py`

- `ori_imgs` is skipped in GPU conversion.
- `np.object_` arrays raise explicit key-aware errors for easier triage.

### 5) Checkpoint loading compatibility

File: `pcdet/models/detectors/detector3d_template.py`

- Uses `torch.load(..., weights_only=False)` when supported (PyTorch 2.6+ behavior compatibility).
- Fallback for older torch versions preserved.

## New Configs (Waymo eval)

- `tools/cfgs/dataset_configs/simbev2waymo_dataset.yaml`
- `tools/cfgs/simbev2waymo_models/unitr.yaml`
- `tools/cfgs/simbev2waymo_models/unitr+lss.yaml`

Key settings:

- `DATA_PATH: ../data/simbev/setup/waymo_val`
- `INFO_PATH.test: [infos/simbev_infos_val.json]`
- `CAMERA_CONFIG.IMAGE.PAD_TO_MAX_SHAPE: True`
- `CAMERA_CONFIG.IMAGE.PAD_ALIGN: center`
- `LIDAR2IMAGE.lidar2image_layer.sparse_shape: [96, 264, 5]` (5-camera Waymo setup)

## Recommended Eval Commands

Run from `tools/`:

```bash
python test.py \
  --cfg_file cfgs/simbev2waymo_models/unitr.yaml \
  --ckpt ../checkpoints/unitr-det.pth \
  --launcher none \
  --workers 0 \
  --batch_size 1
```

```bash
bash scripts/dist_test.sh 1 \
  --cfg_file cfgs/simbev2waymo_models/unitr.yaml \
  --ckpt ../checkpoints/unitr-det.pth
```

For UniTR+LSS:

```bash
bash scripts/dist_test.sh 1 \
  --cfg_file cfgs/simbev2waymo_models/unitr+lss.yaml \
  --ckpt ../checkpoints/unitr-det.pth
```

With debug:

```bash
UNITR_DEBUG_IMAGE2LIDAR=1 bash scripts/dist_test.sh 1 \
  --cfg_file cfgs/simbev2waymo_models/unitr.yaml \
  --ckpt ../checkpoints/unitr-det.pth
```

## Notes

- Worker `BrokenPipe` / `Aborted` can be secondary symptoms after an upstream model exception.
- If switching datasets frequently, prefer `--set DATA_CONFIG.DATA_PATH ...` for temporary overrides.
