#!/bin/bash
set -euo pipefail

#Init absolute path
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path/.."
mkdir -p data/raw_data

# Define arrays of parameters to test
seeds=(900)
num_robots=(40)
agent_capacity=2
task_weight=1
time_horizons=(10)
max_tasks=(120)
frequencies=(0.25)
inbound_outbound_ratio=(1.0)
num_skus=(30)
initial_inventory=(30.0)
weight_init_method="uniform"
task_gen_strategy="feedback_control"
initial_task_assign_strategy="fast_greedy"
improvement_task_assign_strategy="cbta_dcbs"
cost_calculation_method="shortest_path"
path_planning_strategy="dcbs"
map="data/maps/study_small_restricted"
removal_operator="shaw"
repair_operator="greedy"
acceptance_function="simulated_annealing"
T_0=1.0
alpha=0.99
# Use "none" for tasks without deadlines; otherwise constant/normal/bimodal.
deadline_generation_method="normal"
deadline_offset=180
base_cost_weight=1.0
# 0 disables deadline cost; a positive value penalizes estimated lateness.
# With deadline_generation_method="none", CBTA ignores this penalty entirely.
deadline_weight=0.0
sku_distribution_weight=0.0
agent_unallocated_penalty=5.0

# Loop through all combinations
for seed in "${seeds[@]}"; do
    for robots in "${num_robots[@]}"; do
        for T in "${time_horizons[@]}"; do
            for max_task in "${max_tasks[@]}"; do
                for frequency in "${frequencies[@]}"; do
                    for num_sku in "${num_skus[@]}"; do
                        echo "Running experiment with:"
                        echo "  Seed: $seed"
                        echo "  Robots: $robots"
                        echo "  Agent Capacity: $agent_capacity"
                        echo "  Task Weight: $task_weight"
                        echo "  Time Horizon: $T"
                        echo "  Max Tasks: $max_task"
                        echo "  Frequency: $frequency"
                        echo "  Inbound Outbound Ratio: $inbound_outbound_ratio"
                        echo "  Number of SKUs: $num_sku"
                        echo "  Weight Init Method: $weight_init_method"
                        echo "  Task Gen Strategy: $task_gen_strategy"
                        echo "  Initial Task Assign Strategy: $initial_task_assign_strategy"
                        echo "  Improvement Task Assign Strategy: $improvement_task_assign_strategy"
                        echo "  Path Planning Strategy: $path_planning_strategy"
                        echo "  Map: $map"
                        echo "  Removal Operator: $removal_operator"
                        echo "  Repair Operator: $repair_operator"
                        echo "  Acceptance Function: $acceptance_function"
                        echo "  T_0: $T_0"
                        echo "  Alpha: $alpha"
                        echo "  Base Cost Weight: $base_cost_weight"
                        echo "  Deadline Weight: $deadline_weight"
                        echo "  Sku Distribution Weight: $sku_distribution_weight"
                        echo "  Agent Unallocated Penalty: $agent_unallocated_penalty"
                        echo "----------------------------------------"
                        if (( $# )); then
                            echo "Command-line overrides: $*"
                        fi
                        
                        "${PYTHON:-python3}" "$parent_path/GT_grid_world.py" \
                            --seed "$seed" \
                            --num-robots "$robots" \
                            --agent-capacity "$agent_capacity" \
                            --task-weight "$task_weight" \
                            --time-horizon "$T" \
                            --max-tasks "$max_task" \
                            --task-gen-strategy "$task_gen_strategy" \
                            --initial-task-assign-strategy "$initial_task_assign_strategy" \
                            --improvement-task-assign-strategy "$improvement_task_assign_strategy" \
                            --path-planning-strategy "$path_planning_strategy" \
                            --time-limit 86400 \
                            --initial-inventory "$initial_inventory" \
                            --frequency "$frequency" \
                            --inbound-outbound-ratio "$inbound_outbound_ratio" \
                            --num-skus "$num_sku" \
                            --weight-init-method "$weight_init_method" \
                            --map "$map" \
                            --cost-calculation-method "$cost_calculation_method" \
                            --removal-operator "$removal_operator" \
                            --repair-operator "$repair_operator" \
                            --acceptance-function "$acceptance_function" \
                            --T-0 "$T_0" \
                            --alpha "$alpha" \
                            --deadline-generation-method "$deadline_generation_method" \
                            --deadline-offset "$deadline_offset" \
                            --base-cost-weight "$base_cost_weight" \
                            --deadline-weight "$deadline_weight" \
                            --sku-distribution-weight "$sku_distribution_weight" \
                            --agent-unallocated-penalty "$agent_unallocated_penalty" \
                            "$@"

                        # Optional: Add a small delay between runs
                        sleep 1
                    done
                done
            done
        done
    done
done

echo "All experiments completed!"
