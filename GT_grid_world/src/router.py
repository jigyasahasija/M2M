from src.path_finding_algorithms.external_algorithms.EECBS import eecbs
from src.path_finding_algorithms.external_algorithms.PBS import pbs

from .agent import *
from .utils import *
from .analysis.statistics import *
from typing import Dict, Tuple
from .graph import Graph
from .path_finding_algorithms.dcbs import plan_dcbs

def pathPlan(map : str, Rs : AgentLoader, path_planning_strategy : str, S : Stats,
             G: Graph = None) -> AgentLoader:
    """Dispatch routing to DCBS, PBS, or EECBS.

    DCBS requires the simulator's graph and returns one action per robot.
    PBS/EECBS retain their existing map-based interface and full paths.
    """
    if path_planning_strategy == "dcbs":
        if G is None:
            raise ValueError("DCBS routing requires the simulator graph G")
        return plan_dcbs(G, Rs, S)

    goal_locations = []
    for agent in Rs.agents:
        # If robot is going to pickup, set goal location to the task's start location
        if agent.status == 1:
            # Get current assigned task's start location
            goal_locations.append(agent.task_sequence[0][1])
            
        # If robot is going to delivery, set goal location to the task's goal location
        elif agent.status == 2:
            goal_locations.append(agent.task_sequence[0][2])
            
        # If robot is a free_agent, set goal location to current state
        else:
            goal_locations.append(agent.state)

    sequences = []
    w = 1.2

    # If two agents have the same goal location, set the goal location of the agnet with no task to its home location
    # Check if any goal location is found more than once
    if len(goal_locations) != len(set(goal_locations)):
        # raise ValueError(f"Agents have duplicate goal locations: {goal_locations} with agent states: {states}")
        for i, agent in enumerate(Rs.agents):
            if goal_locations.count(goal_locations[i]) > 1 and agent.status == 0:
                goal_locations[i] = agent.home

    print(f"Goal locations: {goal_locations}")
    
    latch = False
    while not sequences:
        # Execute the path planning algorithm
        if path_planning_strategy == "pbs":
            sequences = pbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        elif path_planning_strategy == "ecbs":
            sequences = eecbs.test_cpp_func(map, len(Rs.agents), 1, w, Rs.get_agent_states(), goal_locations)
        if sequences == []:
            print("+++++++++++++++++++Execution Failed with w = ", w)
            S.update_num_path_plan_fail()
            
        # If a solution cannot be found with a higher suboptimality bound, break
        if w >= 1.2:
            if latch:
                break
            for i, agent in enumerate(Rs.agents):
                if agent.path_sequence == []:
                    goal_locations[i] = agent.home
            latch = True
        w += 5.0

    if not sequences:
        return Rs
    
    # Remove first item in sequences, as they are the robot's current location
    for i, agent in enumerate(Rs.agents):
        temp_sequence = sequences[i][1:]
        agent.path_sequence = temp_sequence
    
    return Rs
