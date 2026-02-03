# Changes to train.py

## Overview
This is the fully reconstructed and cleaned up `train.py` with all improvements from our development sessions.

## Key Features Added

### 1. **Intrinsic Loss Branch (Ordinal Albedo)**
- Flag: `--use-intrinsic` - Enable intrinsic-image L1 loss based on ordinal albedo
- Flag: `--intrinsic-lambda` (default 0.1) - Weight for intrinsic loss
- Flag: `--intrinsic-every` (default 10) - Runs intrinsic loss every N iterations
- Flag: `--intrinsic-scale-mode` {avg|none} - Scaling mode:
  - `avg` (default): multiply by N to keep average strength unchanged
  - `none`: no scaling applied

**How it works:**
- When enabled, the intrinsic loss runs every `intrinsic_every` steps
- Uses ordinal albedo models (v2) from `intrinsic.pipeline`
- Compares predicted albedo with current Kd texture
- Automatically scales loss to maintain consistent average strength

### 2. **Benchmarking System**
- Flag: `--bench` - Enable per-iteration benchmarking
- Flag: `--bench-start` (default 100) - Start logging at this iteration
- Flag: `--bench-output` (default 'bench.jsonl') - Output JSONL per iter
- Flag: `--bench-summary` (default 'bench_summary.json') - Final summary JSON

**Metrics collected:**
- Timing: iter_ms, data_ms, fwd_ms, bwd_ms, opt_ms, clamp_ms
- Throughput: throughput_mpix_s, images_per_s
- Memory: max_mem_gb, reserved_gb
- Losses: img_loss, reg_loss, intr_loss
- Learning rate: lr

**Output:**
- `bench.jsonl`: One JSON object per line, one per tracked iteration
- `bench_summary.json`: Aggregate statistics (count, mean, median, min, max) for each metric
- Console: Prints median iter time, throughput, device info

### 3. **Enhanced Trainer Class**
- Now manages intrinsic loss pipeline internally
- Supports loading ordinal models once per training session
- Tracks `last_intr_loss` for logging
- Parameters: `use_intrinsic`, `lambda_intr`, `intrinsic_every`, `scale_mode`

### 4. **Safer Material/Texture Saving**
- Display block now safely checks if 'kd' exists in opt_material
- Falls back to result_dict['kd'] if 2D texture not available (e.g., during MLP phase)
- Prevents crashes when Pass 1 uses MLP-based materials

### 5. **Auto Camera Radius from Mesh**
- When loading .obj files:
  - Computes bounding box of reference mesh
  - Sets RADIUS = max(2.0, diagonal * 2.0)
  - Automatically adjusts for model size

### 6. **Fixed & Improved validate() Function**
- No leaking references to training dataloaders
- Properly creates dedicated validation dataloader
- Returns averaged PSNR value
- Writes metrics.txt with format: ID, MSE, PSNR + AVERAGES line
- Supports checkpoint saving during validation

### 7. **Improved validate_itr()**
- Simplified display layer logic
- Safely handles relight and bsdf layers
- Better organization of result handling

## Command-Line Arguments (New)

```bash
# Intrinsic loss controls
--use-intrinsic              Enable intrinsic-image (ordinal albedo) L1 loss
--intrinsic-lambda FLOAT     Weight (default 0.1)
--intrinsic-every INT        Run every N iters (default 10)
--intrinsic-scale-mode STR   'avg' or 'none' (default 'avg')

# Benchmarking
--bench                      Enable per-iter benchmarking
--bench-start INT            Start at iteration (default 100)
--bench-output STR           JSONL filename (default 'bench.jsonl')
--bench-summary STR          Summary filename (default 'bench_summary.json')

# Other
--seed INT                   RNG seed (default 42)
```

## Training Workflow

### Two-Pass Default:
1. **Pass 1 (dmtet_pass1)**: SDF isosurface with MLP material
   - Uses intrinsic loss if `--use-intrinsic` enabled
   - Outputs to `out_dir/dmtet_validate/` and `dmtet_mesh/`

2. **Pass 2 (mesh_pass)**: Fixed-topology mesh refinement
   - Uses intrinsic loss if `--use-intrinsic` enabled
   - Outputs to `out_dir/validate/` and `mesh/`

### Single-Pass Mode (with `--base-mesh`):
- Load existing mesh and refine topology
- Single pass with intrinsic loss support

## Logging Output

Default logging every 10 iterations:
```
iter= 0, img_loss=0.123456, reg_loss=0.009876, intrinsc_loss=0.000000, lr=0.00150, time=234.5 ms, rem=48h 23m
```

With benchmarking enabled, also produces:
- `bench.jsonl`: Detailed per-iteration metrics
- `bench_summary.json`: Statistical summary with system info and flags

## Testing the Changes

### Verify intrinsic loss (every 10 steps, scaled mode):
```bash
python train.py \
  -rm data/my_mesh.obj \
  -o test_intrinsic \
  -i 100 \
  --use-intrinsic \
  --intrinsic-lambda 0.1 \
  --intrinsic-every 10 \
  --intrinsic-scale-mode avg
```

### Enable benchmarking:
```bash
python train.py \
  -rm data/my_mesh.obj \
  -o test_bench \
  -i 100 \
  --bench \
  --bench-start 20
```

### Disable intrinsic (baseline):
```bash
python train.py \
  -rm data/my_mesh.obj \
  -o test_baseline \
  -i 100
```

## File Structure

```
/home/elghobashy/ghobashy/nvdiffrec/
├── train.py (✓ Updated)
├── train.py.bak (original backup)
├── configs/ (configuration files)
├── geometry/ (geometry modules)
├── render/ (rendering modules)
├── dataset/ (dataset loaders)
├── intrinsic/ (new - ordinal albedo pipeline)
└── out/ (output directory)
```

## Notes

- RADIUS default changed from 2.0 to 6.0
- Intrinsic pipeline requires models in `intrinsic/` directory
- Benchmarking adds minimal overhead when disabled
- All existing functionality preserved
- Compatible with multi-GPU training (distributed)
