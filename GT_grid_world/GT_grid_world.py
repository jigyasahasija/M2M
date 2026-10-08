import time
import numpy as np
import argparse
import math

from src import graph, simulate, task_allocation, case_request_generator, router, agent
from src.analysis import statistics

def execute(S : statistics.Stats, map : str, Rs : agent.AgentLoader, G : graph.Graph, frequency : float, inbound_to_outbound_ratio: float, 
            T: int, case_request_strategy: str = "uninformed_uniform", 
            max_task_number : int = 20,
            initial_task_assignment_strategy : str = "lns",
            improvement_task_assignment_strategy : str = "M2M",
            path_planning_strategy : str = "ecbs", time_limit : int = 999999,
            cost_calculation_method : str = "manhattan",
            removal_operator : str = "worst", repair_operator : str = "greedy",
            acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
            deadline_generation_method: str = "constant",
            deadline_offset: float = 30,
            base_cost_weight: float = 1.0,
            deadline_weight: float = 0.0,
            sku_distribution_weight: float = 0.0,
            agent_unallocated_penalty: float = 0.0,
            initial_inventory : float = 25.0):
    # Initilize empty dict of tasks, task is defined as (id: (start_loc, goal_loc, deadline, sku_id, inbound))
    J = {}

    last_task_id = 0
    
    global_tik = time.time()
    for t in range(T):
        print("============================= T : " + str(t) + "=============================")
        
        print("=============================" + "Task Generation"+ "=============================")
        if t%frequency == 0:
            if len(J) < max_task_number:
                if frequency >= 1.: 
                    N = 1
                elif frequency < 1.: 
                    N = frequency**-1
                else:
                    N = 0
                # Generate new tasks
                tik = time.time()

                J, last_task_id, __, __ = case_request_generator.CRG(S, t, J, G, Rs, N, 
                                                                                                inbound_to_outbound_ratio, last_task_id, max_task_number,
                                                                                                  G.warehouse, case_request_strategy, 
                                                                                                  deadline_generation_method, deadline_offset, improvement_task_assignment_strategy,
                                                                                                  initial_inventory)
                tok = time.time()
                S.add_total_CRG_time(tok-tik)
            
        tik = time.time()

        print("=============================" + "Task Allocation"+ "=============================")
        # Check if all tasks are allocated, if so, skip
        total = 0
        for agent in Rs.agents:
            total += len(agent.task_sequence)
            
        if total < max_task_number:
            print(f"Attempting to allocate tasks")
            Rs, _, _ = task_allocation.TaskAllocation(S, G, Rs, J, initial_task_assignment_strategy, improvement_task_assignment_strategy, map, t, cost_calculation_method, removal_operator, repair_operator, acceptance_function, T_0, alpha, base_cost_weight, deadline_weight, sku_distribution_weight, agent_unallocated_penalty)

        tok = time.time()
        S.add_total_TA_time(tok-tik)

        # Check if any agent is allocated the same tasks
        for agent in Rs.agents:
            task_ids = set()
            for task_id in agent.task_sequence:
                if task_id in task_ids:
                    raise ValueError(f"Task {task_id} is allocated to multiple agents.")
                task_ids.add(task_id)

        
        print("=============================" +"Routing"+ "=============================")
        tik = time.time()     
        
        # DCBS replans every tick; the other planners refresh exhausted paths.
        if path_planning_strategy == "dcbs" or any(not agent.path_sequence for agent in Rs.agents):
            Rs = router.pathPlan(map, Rs, path_planning_strategy, S, G=G)

        tok = time.time()
        S.add_total_PF_time(tok-tik)
        
        soc = 0
        for agent in Rs.agents:
            soc += len(agent.path_sequence)
        S.set_soc(soc)
        
        print("=============================" +"Taking Step"+ "=============================")
        tik = time.time()
        Rs, J = simulate.simulate(S, G, Rs, J, map, t)
        tok = time.time()
        S.add_total_SIM_time(tok-tik)
        
        # Log the number of tasks in the system
        S.append_tasks_in_system(len(J))
        
        # Log warehouse inventory state
        S.append_warehouse_inventory_state(G.warehouse, G.get_aisle_locations())
        
        # Log driveway inventory state
        S.append_driveway_inventory_state(G.driveway)
        
        # Log SKU inventory state
        S.append_sku_inventory_state(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())
        
        # Log SKU centroids per timestep
        S.append_sku_centroids(G.warehouse, G.warehouse.get_all_skus().__len__())
        
        # Log SKU locations per timestep
        S.append_sku_locations(G.warehouse, G.driveway, G.warehouse.get_all_skus().__len__())
        
        # Record agent statuses and goal locations for this timestep
        S.add_agent_statuses_and_goals(Rs)
        
        S.compute_unallocated_agents(Rs)
        
    return

def main(seed: int, num_robots: int, T: int, max_number_tasks: int, 
         task_generation_strategy: str, initial_task_assignment_strategy: str, improvement_task_assignment_strategy: str,
         path_planning_strategy: str, map_name: str, time_limit: int = 999999,
         initial_inventory: float = 25.0, 
         frequency: float = 1.0, inbound_outbound_ratio: float = 1.0,
         num_skus: int = 10,
         weight_init_method: str = "random",
         cost_calculation_method: str = "manhattan",
         removal_operator: str = "worst", repair_operator: str = "greedy",
         acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
         deadline_generation_method: str = "constant",
         deadline_offset: float = 30,
         base_cost_weight: float = 1.0,
         deadline_weight: float = 0.0,
         sku_distribution_weight: float = 0.0,
         agent_unallocated_penalty: float = 0.0, agent_capacity: float = 1, task_weight: float = 1) -> None:
    """
    Run a single instance of the simulation with specified parameters.
    
    Args:
        seed: Random seed for reproducibility
        num_robots: Number of robots in the simulation
        T: Time horizon
        max_number_tasks: Maximum number of tasks
        task_generation_strategy: Strategy for generating tasks
        initial_task_assignment_strategy: Strategy for assigning tasks
        improvement_task_assignment_strategy: Strategy for improving tasks
        path_planning_strategy: Strategy for path planning
        map_name: Name of the map to use
        time_limit: Maximum runtime in seconds
        initial_inventory: Initial inventory fill percentage (0.0 to 100.0)
        frequency: Frequency of task generation
        inbound_outbound_ratio: Ratio of inbound to outbound tasks
        num_skus: Number of unique SKUs in the warehouse
        weight_init_method: Method for initializing SKU weights ("random" or "uniform")
        cost_calculation_method: Method for calculating cost ("manhattan" or "shortest_path")
        removal_operator: Removal operator for task allocation
        repair_operator: Repair operator for task allocation
        acceptance_function: Acceptance function for LNS (greedy or simulated_annealing)
        T_0: Initial temperature for simulated annealing
        alpha: Temperature decay rate for simulated annealing
        deadline_generation_method: Method for generating task deadlines (e.g., constant, normal, bimodal, etc.)
        deadline_offset: Offset for task deadlines
        base_cost_weight: Weight for base cost
        deadline_weight: Weight for deadline
        sku_distribution_weight: Weight for sku distribution
        agent_unallocated_penalty: Weight for agent unallocated penalty
        agent_capacity: Maximum reserved and carried weight for CBTA
        task_weight: Positive weight of each generated task
    """
    if not math.isfinite(task_weight) or task_weight <= 0:
        raise ValueError("task_weight must be finite and positive")
    if task_weight != 1 and improvement_task_assignment_strategy != "cbta_dcbs":
        raise ValueError("Weighted tasks currently require cbta_dcbs")
    if not math.isfinite(agent_capacity) or agent_capacity <= 0:
        raise ValueError("agent_capacity must be positive")
    if agent_capacity != 1 and improvement_task_assignment_strategy != "cbta_dcbs":
        raise ValueError("Multi-item capacity currently requires cbta_dcbs")
    if improvement_task_assignment_strategy == "cbta_dcbs":
        path_planning_strategy = "dcbs"
    elif path_planning_strategy == "dcbs":
        raise ValueError("dcbs routing requires cbta_dcbs task assignment")
    np.random.seed(seed)
    
    stripped_map_name = map_name.split("/")[-1].replace(".json", "")
    
    output_file = f"data/raw_data/{T}_{task_generation_strategy}_{initial_inventory}_{initial_task_assignment_strategy}_{improvement_task_assignment_strategy}_{path_planning_strategy}_{stripped_map_name}_{num_robots}_{max_number_tasks}_{base_cost_weight}_{deadline_weight}_{sku_distribution_weight}_{seed}_capacity{agent_capacity:g}_weight{task_weight:g}.json"

    S = statistics.Stats(
        num_robots=num_robots,
        agent_capacity=agent_capacity,
        task_weight=task_weight,
        simulation_time=T,
        output_file=output_file,
        map_name=map_name,
        cost_calculation_method=cost_calculation_method,
        seed=seed,
        max_tasks=max_number_tasks,
        task_generation_strategy=task_generation_strategy,
        initial_task_assignment_strategy=initial_task_assignment_strategy,
        improvement_task_assignment_strategy=improvement_task_assignment_strategy,
        path_planning_strategy=path_planning_strategy,
        time_limit=time_limit,
        initial_inventory=initial_inventory,
        frequency=frequency,
        inbound_outbound_ratio=inbound_outbound_ratio,
        num_skus=num_skus,
        weight_init_method=weight_init_method,
        removal_operator=removal_operator,
        repair_operator=repair_operator,
        acceptance_function=acceptance_function,
        T_0=T_0,
        alpha=alpha,
        deadline_generation_method=deadline_generation_method,
        deadline_offset=deadline_offset,
        base_cost_weight=base_cost_weight,
        deadline_weight=deadline_weight,
        sku_distribution_weight=sku_distribution_weight,
        agent_unallocated_penalty=agent_unallocated_penalty
    )
    G = graph.Graph(num_robots, map_name, initial_inventory, num_skus, weight_init_method)

    robots = []
    for robot_id, location in enumerate(G.get_all_occupied()):
        robots.append(agent.Agent(robot_id, location, capacity=agent_capacity))
    
    Rs = agent.AgentLoader(robots)
    
    init_locations = [agent.state for agent in Rs.agents]
    S.add_paths(init_locations)
    
    tik = time.time()
    # Execute online algorithm
    execute(S, map_name, Rs, G, frequency, inbound_outbound_ratio, T, 
            case_request_strategy=task_generation_strategy, 
            max_task_number=max_number_tasks, 
            initial_task_assignment_strategy=initial_task_assignment_strategy, 
            improvement_task_assignment_strategy=improvement_task_assignment_strategy, 
            path_planning_strategy=path_planning_strategy, 
            time_limit=time_limit,
            cost_calculation_method=cost_calculation_method,
            removal_operator=removal_operator,
            repair_operator=repair_operator,
            acceptance_function=acceptance_function,
            T_0=T_0,
            alpha=alpha,
            deadline_generation_method=deadline_generation_method,
            deadline_offset=deadline_offset,
            base_cost_weight=base_cost_weight,
            deadline_weight=deadline_weight,
            sku_distribution_weight=sku_distribution_weight,
            agent_unallocated_penalty=agent_unallocated_penalty,
            initial_inventory=initial_inventory
    )
    tok = time.time()
    S.set_total_runtime(tok-tik)
    
    S.save_data()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Run grid world simulation with specified parameters')
    
    parser.add_argument('--agent-capacity', type=float, default=1, help='Weight capacity per robot; nondefault values require cbta_dcbs')
    parser.add_argument('--task-weight', type=float, default=1, help='Positive weight per generated task (CBTA); defaults to 1')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    parser.add_argument('--num-robots', type=int, required=True, help='Number of robots')
    parser.add_argument('--time-horizon', type=int, required=True, help='Time horizon T')
    parser.add_argument('--max-tasks', type=int, required=True, help='Maximum number of tasks')
    parser.add_argument('--task-gen-strategy', type=str, required=True, 
                       choices=['informed_uniform', 'uninformed_uniform', 'feedback_control'],
                       help='Task generation strategy')
    parser.add_argument('--initial-task-assign-strategy', type=str, required=True,
                       choices=['cost_matrix', 'random', 'greedy', 'randomized_greedy', 'FCF', 'max_regret_FC', 'randomized_max_regret_FC', 'fast_greedy', 'fast_FCF', 'fast_SCF'],
                       help='Task assignment strategy')
    parser.add_argument('--improvement-task-assign-strategy', type=str, required=True,
                       choices=['M2M', 'LNS_PBS', 'cbta_dcbs', 'none'],
                       help='Task assignment strategy for improvement')
    parser.add_argument('--path-planning-strategy', type=str, required=True,
                       choices=['ecbs', 'pbs', 'dcbs'],
                       help='Path planning strategy')
    parser.add_argument('--map', type=str, default='data/maps/symbotic_small',
                       help='Map file path')
    parser.add_argument('--time-limit', type=int, default=999999,
                       help='Maximum runtime in seconds')
    parser.add_argument('--initial-inventory', type=float, default=25.0,
                       help='Initial inventory fill percentage (0.0 to 100.0)')
    parser.add_argument('--frequency', type=float, default=1.0,
                       help='Frequency of task generation')
    parser.add_argument('--inbound-outbound-ratio', type=float, default=1.0,
                       help='Ratio of inbound to outbound tasks')
    parser.add_argument('--num-skus', type=int, default=10,
                       help='Number of unique SKUs in the warehouse')
    parser.add_argument('--weight-init-method', type=str, default='random',
                       choices=['random', 'uniform'],
                       help='Method for initializing SKU weights')
    parser.add_argument('--cost-calculation-method', type=str, default='manhattan',
                       choices=['manhattan', 'shortest_path'],
                       help='Method for calculating cost')
    parser.add_argument('--removal-operator', type=str, default='worst',
                       choices=['worst', 'random', 'greedy', 'shaw'],
                       help='Removal operator for task allocation')
    parser.add_argument('--repair-operator', type=str, default='greedy',
                       choices=['greedy', 'random', 'worst', 'fast_SCF'],
                       help='Repair operator for task allocation')
    parser.add_argument('--acceptance-function', type=str, default='greedy', choices=['greedy', 'simulated_annealing'], help='Acceptance function for LNS (greedy or simulated_annealing)')
    parser.add_argument('--T-0', type=float, default=1.0, help='Initial temperature for simulated annealing')
    parser.add_argument('--alpha', type=float, default=0.99, help='Temperature decay rate for simulated annealing')
    parser.add_argument('--deadline-generation-method', type=str, default='constant',
                       help='Task deadline generation; none disables deadlines', choices=['constant', 'normal', 'bimodal', 'none'])
    parser.add_argument('--deadline-offset', type=float, default=30, help='Offset for task deadlines')
    parser.add_argument('--base-cost-weight', type=float, default=1.0, help='Weight for base cost')
    parser.add_argument('--deadline-weight', type=float, default=0.0, help='CBTA lateness cost weight: 0 disables the penalty; positive values enable it when tasks have deadlines')
    parser.add_argument('--sku-distribution-weight', type=float, default=0.0, help='Weight for sku distribution')
    parser.add_argument('--agent-unallocated-penalty', type=float, default=0.0, help='Penalty for unallocated agents')
    args = parser.parse_args()
    
    main(
        agent_capacity=args.agent_capacity,
        task_weight=args.task_weight,
        seed=args.seed,
        num_robots=args.num_robots,
        T=args.time_horizon,
        max_number_tasks=args.max_tasks,
        task_generation_strategy=args.task_gen_strategy,
        initial_task_assignment_strategy=args.initial_task_assign_strategy,
        improvement_task_assignment_strategy=args.improvement_task_assign_strategy,
        path_planning_strategy=args.path_planning_strategy,
        map_name=args.map,
        time_limit=args.time_limit,
        initial_inventory=args.initial_inventory,
        frequency=args.frequency,
        inbound_outbound_ratio=args.inbound_outbound_ratio,
        num_skus=args.num_skus,
        weight_init_method=args.weight_init_method,
        cost_calculation_method=args.cost_calculation_method,
        removal_operator=args.removal_operator,
        repair_operator=args.repair_operator,
        acceptance_function=args.acceptance_function,
        T_0=args.T_0,
        alpha=args.alpha,
        deadline_generation_method=args.deadline_generation_method,
        deadline_offset=args.deadline_offset,
        base_cost_weight=args.base_cost_weight,
        deadline_weight=args.deadline_weight,
        sku_distribution_weight=args.sku_distribution_weight,
        agent_unallocated_penalty=args.agent_unallocated_penalty,
    )
