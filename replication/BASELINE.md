# Replication baseline

This directory records the environment and inputs used for the MLAPD paper
replication. It is separate from the existing `GT_grid_world` simulator, which
is referred to as the **repo baseline**.

## Frozen starting point

- Top-level commit: `afc8ce0488a6f3d7e912696e41f3b5e02893959b`
- Python: CPython 3.12.12, created with `uv`
- Python dependencies: [requirements.lock](../requirements.lock)
- Submodules: EECBS `73248c53306c3cbfa2b4df7b9202b0e5de753336`, PBS
  `d8417a17b1bb1966829013a9a392703913aec173`, and LNS
  `3e0961f5592e4c0310274115d0602c19a15d2a04`

Python 3.11 is not suitable for this lockfile because the pinned
`numpy==2.5.1` requires Python 3.12 or newer.

## Repo-baseline smoke test

The original simulator completed this short EECBS run on `data/maps/small_test`:

```bash
.venv/bin/python GT_grid_world/GT_grid_world.py \
  --seed 1 --num-robots 1 --time-horizon 3 --max-tasks 1 \
  --task-gen-strategy feedback_control \
  --initial-task-assign-strategy fast_greedy \
  --improvement-task-assign-strategy none \
  --path-planning-strategy ecbs --map data/maps/small_test \
  --initial-inventory 20 --frequency 1 --inbound-outbound-ratio 1 \
  --num-skus 1 --weight-init-method uniform \
  --cost-calculation-method manhattan \
  --deadline-generation-method constant --deadline-offset 30
```

This proves that the existing simulator and EECBS extension are runnable. It
does not validate the paper's MLAPD formulation. Paper-specific work lives in
`GT_grid_world/src/mlapd_replication` and consumes immutable trace files.

## Native-module build note

PBS and EECBS required the small CMake compatibility patches in `patches/` to
build with the pinned modern `pybind11`. The extensions were built locally with
CMake, Eigen, Boost, and Dlib installed through Homebrew.
