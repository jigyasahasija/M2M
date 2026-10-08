# M2M

Task allocation and multi-agent path planning simulator for warehouse robotics, developed at Georgia Tech. Paper was published in ICRA 2026: {[M2M](https://arxiv.org/pdf/2605.07835)} and was sponsored by [Symbotic](https://www.symbotic.com/).

---

## Table of Contents

- [Installation](#installation)
  - [Python Requirements](#python-requirements)
  - [C++ Requirements](#c-requirements)
  - [Git Submodules](#git-submodules)
  - [Building the C++ Modules](#building-the-c-modules)
- [Running Experiments](#running-experiments)
  - [run_experiments.sh Parameters](#run_experimentssh-parameters)
- [Paper Experiments](#paper-experiments)

---

## Installation

### Python Requirements

Python 3.11+ is required. Install all Python dependencies with:

```bash
pip install -r requirements.txt
```
or manually install the following packages:

Key packages include `numpy`, `scipy`, `pybind11`, `pybind["global"]`, and `munkres`

### C++ Requirements

The C++ submodules (LNS, C-LNS, PBS, EECBS) require the following system libraries:

```bash
# Ubuntu / Debian
sudo apt-get install cmake build-essential libeigen3-dev libboost-all-dev
```

- **CMake** >= 2.6
- **Eigen3** — linear algebra library used internally by LNS
- **Boost** — `program_options`, `system`, and `filesystem` components
- **pybind11** — installed via pip (included in `requirements.txt`)

### Git Submodules

The three C++ solvers are included as git submodules. After cloning the repo, initialize them with:

```bash
git submodule update --init --recursive
```

This will populate:

| Submodule | Path | Purpose |
|-----------|------|---------|
| LNS | `GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns` | Large Neighborhood Search task allocator |
| PBS | `GT_grid_world/src/path_finding_algorithms/external_algorithms/PBS` | Priority-Based Search path planner |
| EECBS | `GT_grid_world/src/path_finding_algorithms/external_algorithms/EECBS` | Enhanced ECBS path planner |

### Building the C++ Modules

Each submodule must be compiled in-place so that Python can import the resulting `.so` files directly from the submodule directory. Build each one with `cmake` and `make`:

**LNS (task allocation):**
```bash
cd GT_grid_world/src/task_allocation_algorithms/external_algorithms/lns
cmake .
make
```

**PBS (path planning):**
```bash
cd GT_grid_world/src/path_finding_algorithms/external_algorithms/PBS
cmake .
make
```

**EECBS (path planning):**
```bash
cd GT_grid_world/src/path_finding_algorithms/external_algorithms/EECBS
cmake .
make
```

After building, each directory should contain a shared library (e.g., `lns.cpython-*.so`, `pbs.cpython-*.so`) that Python imports at runtime.

---

## Running Experiments

Experiments are launched via the shell script in `GT_grid_world/`:

```bash
cd GT_grid_world
bash run_experiments.sh
```

The script creates output directories (`data/raw_data`, `data/buffer_data`, `data/videos`), then sweeps over all combinations of the specified parameter arrays, invoking `GT_grid_world.py` for each combination.

### run_experiments.sh Parameters

**Sweep arrays** — the script loops over every combination of values in these arrays:

| Parameter | Description |
|-----------|-------------|
| `seeds` | Random seeds for reproducibility |
| `num_robots` | Number of warehouse robots |
| `time_horizons` | Simulation duration in seconds (e.g., `28800` = 8 hours) |
| `max_tasks` | Maximum number of tasks buffered per robot |
| `frequencies` | Task arrival rate (tasks per second) |
| `num_skus` | Number of distinct SKU types in the inventory |

**Fixed parameters** — set once and shared across all runs:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `inbound_outbound_ratio` | `1.0` | Ratio of inbound to outbound tasks |
| `initial_inventory` | `60.0` | Starting inventory level per SKU |
| `weight_init_method` | `uniform` | How SKU weights are initialized |
| `task_gen_strategy` | `feedback_control` | Task generation strategy; uses inventory feedback to regulate task rate |
| `initial_task_assign_strategy` | `fast_greedy` | Algorithm used for the initial task assignment |
| `improvement_task_assign_strategy` | `cbta_dcbs` | Cost-based task assignment with DCBS routing |
| `cost_calculation_method` | `shortest_path` | How agent travel costs are estimated |
| `path_planning_strategy` | `dcbs` | Dynamic conflict-based routing |
| `map` | `data/maps/study_small_restricted` | Warehouse map file |
| `removal_operator` | `shaw` | LNS removal heuristic (Shaw removal) |
| `repair_operator` | `greedy` | LNS repair heuristic |
| `acceptance_function` | `simulated_annealing` | Solution acceptance criterion |
| `T_0` | `1.0` | Initial temperature for simulated annealing |
| `alpha` | `0.99` | Cooling rate for simulated annealing |
| `deadline_generation_method` | `normal` | Distribution used to sample task deadlines |
| `deadline_offset` | `180` | Mean deadline offset in seconds |
| `base_cost_weight` | `1.0` | Weight on travel cost in the objective |
| `deadline_weight` | `0.0` | Weight on deadline violations in the objective |
| `sku_distribution_weight` | `0.0` | Weight on SKU distribution balance in the objective |
| `agent_unallocated_penalty` | `5.0` | Penalty applied per unallocated agent |

**Output** — results for each run are written to `data/raw_data/`.

### CBTA/DCBS

Run from the repository root (script arguments override its defaults):

```bash
PYTHON="$PWD/.venv/bin/python" bash GT_grid_world/run_experiments.sh \
  --improvement-task-assign-strategy cbta_dcbs --path-planning-strategy dcbs \
  --agent-capacity 2
```

CBTA supports multiple carried items with `--agent-capacity` (script default 2).
Set `--task-weight` for generated tasks (default 1, positive fractions allowed).
Each assigned task reserves its weight until delivery. Both reserved and carried
weight must fit the robot's capacity. Overweight tasks remain unassigned. Pickup and
delivery events can interleave, including items sharing a SKU. With unit weights,
capacity 1 executes tasks serially. Other strategies require capacity and weight 1.
Batch scheduling, SSP assignment, and task packaging still differ from the paper.

Cost uses `base_cost_weight` for travel and `deadline_weight` for lateness.
Set `--deadline-weight 0` to ignore lateness in allocation, or a positive value
to enable it. Set `--deadline-generation-method none` to disable task deadlines
and their penalty entirely.

Results in `data/raw_data/` include throughput, computation times, completion
ratio (completed / generated tasks), and weighted costs for completed tasks.
Assignment cost estimates are recorded separately from realized costs. Output
filenames include capacity and task weight, and JSON records each robot’s load and carried task
IDs per timestep. Travel cost sums per-task travel: a shared move counts for
each carried item, rather than once for the robot.
JSON also records per-task weights and carried/reserved weight per timestep.
Mixed-weight tasks can supply weight as the sixth field of a task in `J`;
legacy five-field tasks weigh 1. Binary assignment optimization keeps each task
whole and maximizes assigned task count before minimizing estimated cost.

Tests: `.venv/bin/python -m unittest discover -s GT_grid_world/tests -v`

---

## 

Codebase to visualize output files from simulation: [M2M_Visualizer](https://github.com/Ethan-Schneider/M2M_Visualizer)

## Paper Experiments

> This section describes the experiments conducted for our paper. Details to be filled in.

<!-- TODO: describe the experimental setup, baselines compared, metrics reported, and how to reproduce the main results from the paper. -->
