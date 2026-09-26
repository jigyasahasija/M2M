# Steps 3–5: baseline experiment layer

This directory contains a **debug-scale independent reproduction**, not a
claim that paper numbers have been reproduced.  The paper's original maps,
task traces, and complete tie-breaking rules are not included in the paper.

## Step 3: conflict-free movement

`src/mlapd_replication/space_time_astar.py` adds time to the standalone A*
search.  It allows an agent to wait, rejects two agents in the same cell at
the same time, and rejects an edge swap in one tick.  Agent IDs provide a
fixed priority order for sequential planning.

## Step 4: paper baselines

- `tpmc`: picks the nearest feasible agent using static A* distance from its
  current position to the pickup.
- `rmca`: picks the feasible agent with the smallest added static distance
  from the tail of its existing schedule through the new task.

Both append a task's pickup and delivery to the schedule.  This clearly
defined tail-insertion rule is needed because the paper does not publish every
baseline tie-break and schedule-order detail.

## Step 5: batch runner and metrics

Run either baseline against the same saved trace from `GT_grid_world`:

```bash
../.venv/bin/python -m src.mlapd_replication.runner \
  --trace ../replication/traces/small_test_seed_1.json \
  --baseline tpmc --batch-size 5 \
  --output ../replication/results/small_test_seed_1_tpmc.json
```

Change `--baseline` to `rmca` for the other baseline.  The result JSON records
each batch's assignments plus:

- `completion_rate`: completed tasks divided by all tasks in the frozen trace;
- `travel_distance`: all non-wait moves made by every agent;
- `delay_cost`: sum of `max(0, completion time - deadline)` for tasks that
  finish within the horizon;
- `total_cost`: `alpha * travel_distance + beta * delay_cost`, with paper
  defaults `alpha = beta = 0.5`;
- `average_batch_path_planning_seconds`: the average routing time at batch
  boundaries.

Use `small_test` only to test the implementation.  It does not match the
paper's 21 by 35 or 101 by 81 maps, so its metrics must not be compared to the
paper's reported figures.
