from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *
from typing import Dict, Tuple
from .task_allocation_algorithms.LNS_PBS import lns_pbs_call
from .task_allocation_algorithms.initial_solutions.max_regret_FC import max_regret_FC_call
from .task_allocation_algorithms.initial_solutions.randomized_max_regret_FC import randomized_max_regret_FC_call
from .task_allocation_algorithms.initial_solutions.FCF import FCF_call
from .task_allocation_algorithms.initial_solutions.greedy_allocation import greedy_call
from .task_allocation_algorithms.initial_solutions.randomized_greedy import randomized_greedy_call
from .task_allocation_algorithms.initial_solutions.random_allocation import random_call  
from .task_allocation_algorithms.M2M import M2M_call
from .task_allocation_algorithms.initial_solutions.fast_FCF import fast_FCF_call
from .task_allocation_algorithms.initial_solutions.fast_SCF import fast_SCF_call
from .task_allocation_algorithms.initial_solutions.fast_greedy import fast_greedy_call
from .task_allocation_algorithms.cbta_dcbs import cbta_dcbs_call

def TaskAllocation(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], initial_task_assignment_strategy : str, 
                   improvement_task_assignment_strategy : str, map : str, t : int, cost_calculation_method : str, 
                   removal_operator : str = "worst", repair_operator : str = "greedy", 
                   acceptance_function: str = "greedy", T_0: float = 1.0, alpha: float = 0.99,
                   base_cost_weight: float = 1.0,
                   deadline_weight: float = 0.0,
                   sku_distribution_weight: float = 0.0,
                   agent_unallocated_penalty: float = 0.0) -> AgentLoader:
    """ Task allocation entrance function, which calls the respsective task assignment algorithm and returns the updated task assignment, set of free_agents, and set of to_pickup agents.

    Args:
        S (Stats): Statistics object
        G (Graph): Map data object
        Rs (list): Agent Loader Object
        J (dict): Dict of tasks in the form {task_id: (start_loc, goal_loc, deadline, sku_id, inbound)}
        initial_task_assignment_strategy (str): Chosen task_allocation algorithm to use
        improvement_task_assignment_strategy (str): Chosen task_allocation algorithm to use
        map (str): Map name
        t (int): Current timestep
        cost_calculation_method (str): Method for calculating cost ("manhattan" or "shortest_path")
        removal_operator (str): Removal operator for LNS ("random" or "worst")
        repair_operator (str): Repair operator for LNS ("greedy")
        acceptance_function (str): Acceptance function for M2M
        T_0 (float): Initial temperature for M2M
        alpha (float): Cooling rate for M2M

    Returns:
        AgentLoader: Returns updated AgentLoader object
    """

    if improvement_task_assignment_strategy == "M2M":
        return M2M_call(S, G, Rs, J, initial_task_assignment_strategy, time_limit=1.0, removal_size=2, cost_calculation_method=cost_calculation_method, removal_operator=removal_operator, repair_operator=repair_operator, t=t, acceptance_function=acceptance_function, T_0=T_0, alpha=alpha, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_unallocated_penalty=agent_unallocated_penalty)
    elif improvement_task_assignment_strategy == "LNS_PBS":
        return lns_pbs_call(S, G, map, Rs, J, t)
    elif improvement_task_assignment_strategy == "cbta_dcbs":
        return cbta_dcbs_call(S, G, Rs, J, t, base_cost_weight, deadline_weight)
    elif improvement_task_assignment_strategy == "none":
        pass
    else:
        print("ERROR: Unknown task assignment strategy " + improvement_task_assignment_strategy + ", please choose another one.")
        return Rs

    if initial_task_assignment_strategy == "random":
        return random_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "c_lns":
        return lns_pbs_call(S, G, map, Rs, J, t)
    elif initial_task_assignment_strategy == "greedy":
        return greedy_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "randomized_greedy":
        return randomized_greedy_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "FCF":
        return FCF_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "max_regret_FC":
        return max_regret_FC_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "randomized_max_regret_FC":
        return randomized_max_regret_FC_call(S, G, Rs, J, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "fast_FCF":
        return fast_FCF_call(S, G, Rs, J, t, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "fast_SCF":
        return fast_SCF_call(S, G, Rs, J, t, method=cost_calculation_method)
    elif initial_task_assignment_strategy == "fast_greedy":
        return fast_greedy_call(S, G, Rs, J, t, method=cost_calculation_method, base_cost_weight=base_cost_weight, deadline_weight=deadline_weight, sku_distribution_weight=sku_distribution_weight, agent_task_sequence_limit=3)
    else:
        print("ERROR: Unknown task assignment strategy " + initial_task_assignment_strategy + ", please choose another one.")
        return Rs
