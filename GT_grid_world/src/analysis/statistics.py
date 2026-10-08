import numpy as np
import json

from ..utils import *
from ..agent import *

class Stats: 
    def __init__(self, num_robots: int, simulation_time: int, output_file: str, map_name: str, cost_calculation_method: str,
                 seed: int = None, max_tasks: int = None, task_generation_strategy: str = None,
                 initial_task_assignment_strategy: str = None, improvement_task_assignment_strategy: str = None, path_planning_strategy: str = None,
                 time_limit: int = None, initial_inventory: float = None,
                 frequency: float = None, inbound_outbound_ratio: float = None,
                 num_skus: int = None, weight_init_method: str = None, removal_operator: str = None, repair_operator: str = None,
                 acceptance_function: str = None, T_0: float = None, alpha: float = None, deadline_generation_method: str = None,
                 deadline_offset: float = None, base_cost_weight: float = None, deadline_weight: float = None,
                 sku_distribution_weight: float = None, agent_unallocated_penalty: float = None, agent_capacity: float = 1, task_weight: float = 1) -> None:
        # Store input parameters
        self.__generated_task_weight = task_weight
        self.__task_weights = {}
        self.__agent_carried_weights = []
        self.__agent_reserved_weights = []
        self.__agent_capacity = agent_capacity
        self.__agent_loads = []
        self.__agent_carried_tasks = []
        self.__seed = seed
        self.__num_of_robots = num_robots
        self.__T = simulation_time
        self.__max_tasks = max_tasks
        self.__task_generation_strategy = task_generation_strategy
        self.__initial_task_assignment_strategy = initial_task_assignment_strategy
        self.__improvement_task_assignment_strategy = improvement_task_assignment_strategy
        self.__path_planning_strategy = path_planning_strategy
        self.__map_name = map_name
        self.__time_limit = time_limit
        self.__initial_inventory = initial_inventory
        self.__frequency = frequency
        self.__inbound_outbound_ratio = inbound_outbound_ratio
        self.__num_skus = num_skus
        self.__weight_init_method = weight_init_method
        self.__cost_calculation_method = cost_calculation_method
        self.__removal_operator = removal_operator
        self.__repair_operator = repair_operator
        self.__acceptance_function = acceptance_function
        self.__T_0 = T_0
        self.__alpha = alpha
        self.__deadline_generation_method = deadline_generation_method
        self.__deadline_offset = deadline_offset
        self.__base_cost_weight = base_cost_weight
        self.__deadline_weight = deadline_weight
        self.__sku_distribution_weight = sku_distribution_weight
        self.__agent_unallocated_penalty = agent_unallocated_penalty

        self.__output_file = output_file
        
        self.__early_task_ids = []
        
        # Temp SOC
        self.__soc = 0

        # Track number of tasks in the system per timestep
        self.__tasks_in_system = []

        # Task completion timestamps
        self.__task_completion_timestamps = {}  # task_id -> timestep
        self.__task_release_timestamps = {}     # task_id -> timestep
        self.__service_times = {}               # task_id -> service_time
        self.__total_service_time = 0
        self.__average_service_time = 0
        
        # Cost statistics
        self.__total_task_costs = {}           # task_id -> total_duration
        self.__sum_of_costs = 0
        self.__average_task_cost = 0

        # Admissibility
        self.__admisibility = []
        self.__admisibility_differences = []

        # Aisle and Driveway Occupancy
        self.__aisle_occupancy = []
        self.__driveway_occupancy = []
        
        # Number of collisions at each timestep
        self.__collisions = []
        
        self.__throughputs = []
        
        self.__unallocated_agents = []
        
        # Estimated Distance for Task (task_start -> task_goal) in the form (task_id, estimated_distance)
        self.__estimated_distance = {}
        
        # Actual Distance for Task (task_start -> task_goal) in the form (task_id, actual_distance)
        self.__actual_distance = {}
        
        # Estimated duration for Task (interpolate time for (task_start->task_goal)) in the form (task_id, estimated_duration)
        self.__estimated_duration = {}
        
        # Actual duration for Task (time @ task_goal - time @ task_start) in the form (task_id, actual_duration)
        self.__actual_duration = {}
        
        # Estimated distance for (robot_start_location -> task_start_location) in the form (task_id, estimated_distance)
        self.__estimated_pickup_distance = {}
        
        # Actual distance for (robot_start_location -> task_start_location) in the form (task_id, actual_distance)
        self.__actual_pickup_distance = {}
        
        self.__estimated_pickup_duration = {}
        
        self.__actual_pickup_duration = {}

        self.__opened_nodes = []
        self.__expanded_nodes = []
        
        # Sequence of paths for each robot, init with empty array
        self.__truncated_paths = [] 
        self.__paths = []
        for _ in range(num_robots):
            self.__truncated_paths.append([])
            self.__paths.append([])
            
        self.__completed_task_ids = []
        self.__completed_task_details = {}  # task_id -> (start_location, goal_location)
        self.__completed_to_pickup_task_ids = []
        
        # Runtime Stastics: 
        self.__total_runtime = []
        self.__CRG_time = []
        self.__TA_time = []
        self.__PF_time = []
        self.__SIM_time = []
        
        self.__num_path_plan_fails = 0
        
        # Track task reallocations
        self.__task_reallocations = {}  # task_id -> number of times allocated before being worked on
        
        # Track agent statuses and goal locations per timestep
        self.__agent_statuses_per_timestep = []  # List of lists: [timestep][agent_id] = status
        self.__agent_goal_locations_per_timestep = []  # List of lists: [timestep][agent_id] = goal_location
        self.__agent_task_per_timestep = [] # List of lists: [timestep][agent_id] = task_id
        
        self.__warehouse_full_locations_per_timestep = []
        self.__warehouse_row_counts_per_timestep = []
        self.__warehouse_col_counts_per_timestep = []
        
        self.__driveway_full_locations_per_timestep = []
        self.__warehouse_sku_counts_per_timestep = []
        self.__driveway_sku_counts_per_timestep = []
        
        self.__py_lns_logs = []
        
        # Centroids of each SKU per timestep
        self.__sku_centroids_per_timestep = []
        # Locations of each SKU per timestep
        self.__sku_locations_per_timestep = []
        
        # SKU Agents carrying over time
        
        self.__carrying_skus = []
        
        # Deadline tracking
        self.__task_deadlines = {}  # task_id -> deadline
        self.__overdue_task_completions = 0  # Counter for tasks completed after deadline
        self.__assignment_cost_estimates = {}

        self.reallocation_data = {}
    
    def append_carrying_skus(self, skus : list) -> None:
        self.__carrying_skus.append(skus)

    def get_generated_task_weight(self):
        return self.__generated_task_weight

    def record_task_weight(self, task_id, weight):
        self.__task_weights[int(task_id)] = float(weight)

    def record_assignment_cost(self, task_id: int, cost: float) -> None:
        """Store CBTA's estimate at commitment, not a realized execution cost."""
        self.__assignment_cost_estimates[int(task_id)] = float(cost)

    def get_completion_metrics(self) -> dict:
        generated = len(self.__task_release_timestamps)
        completed = len(self.__completed_task_ids)
        ratio = completed / generated if generated else None
        return {
            "total_generated_tasks": generated,
            "total_uncompleted_tasks": generated - completed,
            "completion_ratio": ratio,
            "completion_percentage": 100 * ratio if ratio is not None else None,
        }

    def get_cbta_cost_metrics(self) -> dict:
        """Realized costs cover completed tasks only, including pickup travel.

        Lateness uses the simulator's recorded completion timestep. Waiting
        contributes to lateness but is not counted as travel distance.
        """
        travel_weight = self.__base_cost_weight if self.__base_cost_weight is not None else 1.0
        delay_weight = self.__deadline_weight or 0.0
        details = {}
        for task_id in self.__completed_task_ids:
            travel = (self.__actual_pickup_distance.get(task_id, 0)
                      + self.__actual_distance.get(task_id, 0))
            deadline = self.__task_deadlines.get(task_id)
            lateness = (max(0, self.__task_completion_timestamps[task_id] - deadline)
                        if deadline is not None and self.__deadline_generation_method != "none" else 0)
            details[task_id] = {
                "travel_distance": float(travel),
                "lateness_timesteps": float(lateness),
                "travel_cost": float(travel_weight * travel),
                "deadline_cost": float(delay_weight * lateness),
                "total_cost": float(travel_weight * travel + delay_weight * lateness),
            }
        total = sum(item["total_cost"] for item in details.values())
        return {
            "cost_scope": "completed_tasks_only",
            "completed_task_costs": details,
            "total_completed_travel_distance": sum(item["travel_distance"] for item in details.values()),
            "total_completed_lateness_timesteps": sum(item["lateness_timesteps"] for item in details.values()),
            "total_travel_cost": sum(item["travel_cost"] for item in details.values()),
            "total_deadline_cost": sum(item["deadline_cost"] for item in details.values()),
            "sum_of_costs": total,
            "average_task_cost": total / len(details) if details else 0.0,
            "assignment_cost_estimates": self.__assignment_cost_estimates.copy(),
            "total_estimated_assignment_cost": sum(self.__assignment_cost_estimates.values()),
        }
    
    def get_num_skus(self) -> int:
        return self.__num_skus

    def get_output_file(self) -> str:   
        return self.__output_file

    def compute_unallocated_agents(self, Rs : AgentLoader):
        num = 0
        for agent in Rs.agents:
            if not agent.task_sequence:
                num += 1
                
        self.__unallocated_agents.append(int(num))
        
    def append_early_task_ids(self, task_id : int) -> None:
        self.__early_task_ids.append(task_id)

    def append_open_nodes(self, opened_nodes : int) -> None:
        self.__opened_nodes.append(int(opened_nodes))

    def append_expanded_nodes(self, expanded_nodes : int) -> None:
        self.__expanded_nodes.append(int(expanded_nodes))
        
    def append_number_of_collisions(self, number_of_collisions : int) -> None:
        self.__collisions.append(int(number_of_collisions))
        
    def add_estimated_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_duration[task_id] = estimated_duration
        
    def add_estimated_pickup_duration(self, task_id : int, estimated_duration : float) -> None:
        self.__estimated_pickup_duration[task_id] = estimated_duration
        
    def get_estimated_duration(self, task_id : int) -> float:
        return self.__estimated_duration.get(task_id, 0)

    def get_estimated_pickup_duration(self, task_id : int) -> float:
        return self.__estimated_pickup_duration.get(task_id, 0)
        
    def add_actual_duration(self, task_id : int) -> None:
        self.__actual_duration[task_id] = 0    
    
    def update_actual_duration(self, task_id : int, actual_duration : float) -> None:
        self.__actual_duration[task_id] = actual_duration
        
    def get_actual_duration(self, task_id : int) -> float:
        return self.__actual_duration.get(task_id, 0)
        
    def update_num_path_plan_fail(self) -> None:
        self.__num_path_plan_fails += 1
        
    def remove_actual_duration(self, task_id : int) -> None:
        if task_id in self.__actual_duration:
            del self.__actual_duration[task_id]
        
    def add_actual_pickup_duration(self, task_id : int) -> None:
        self.__actual_pickup_duration[task_id] = 0    
    
    def update_actual_pickup_duration(self, task_id : int, actual_duration : float) -> None:
        self.__actual_pickup_duration[task_id] = actual_duration
        
    def get_actual_pickup_duration(self, task_id : int) -> float:
        return self.__actual_pickup_duration.get(task_id, 0)
        
    def remove_actual_pickup_duration(self, task_id : int) -> None:
        if task_id in self.__actual_pickup_duration:
            del self.__actual_pickup_duration[task_id]
        
    def remove_uncompleted_task_durations(self) -> None:
        completed_tasks = set(self.__completed_task_ids)
        for task_id in list(self.__actual_duration.keys()):
            if task_id not in completed_tasks:
                self.remove_actual_duration(task_id)
        for task_id in list(self.__actual_pickup_duration.keys()):
            if task_id not in completed_tasks:
                self.remove_actual_pickup_duration(task_id)
                
    def remove_early_task_ids(self) -> None:
        for task_id in self.__actual_duration.copy():
            if task_id in self.__early_task_ids:
                del self.__actual_duration[task_id]
        
        for task_id in self.__estimated_duration.copy():
            if task_id in self.__early_task_ids:
                del self.__estimated_duration[task_id]
                
        for task_id in self.__actual_pickup_duration.copy():
            if task_id in self.__early_task_ids:
                del self.__actual_pickup_duration[task_id]
        
        for task_id in self.__estimated_pickup_duration.copy(): 
            if task_id in self.__early_task_ids:
                del self.__estimated_pickup_duration[task_id]
        
    # def add_estimated_inbound_pickup_distance(self, estimated_distance : float) -> None:
    #     self.__estimated_inbound_pickup_distance.append(estimated_distance)
        
    # def add_actual_inbound_pickup_distance(self, actual_distance : float) -> None:
    #     self.__actual_inbound_pickup_distance.append(actual_distance)
        
    # def add_estimated_outbound_pickup_distance(self, estimated_distance : float) -> None:
    #     self.__estimated_outbound_pickup_distance.append(estimated_distance)
    
    # def add_actual_oubound_pickup_distance(self, actual_distance : float) -> None:
    #     self.__actual_outbound_pickup_distance.append(actual_distance)
    
    # ====================== Estimated Task Distance Function
    
    def add_estimated_distance(self, task_id : int, estimated_distance : float) -> None:
        self.__estimated_distance[task_id] = estimated_distance
 
    # ====================== Estimated Pickup Distance Function
    
    def add_estimated_pickup_distance(self, task_id : int, distance : float) -> None:
        self.__estimated_pickup_distance[task_id] = distance
    
    # ====================== Actual Task Distance Functions
        
    def add_actual_distance(self, task_id : int) -> None:
        self.__actual_distance[task_id] = 0
        
    def update_actual_distance(self, task_id : int, added_distance : float) -> None:
        self.__actual_distance[task_id] += added_distance
        
    def remove_actual_distance(self, task_id : int) -> None:
        del self.__actual_distance[task_id]
        
    def get_actual_distance(self, task_id : int) -> float:
        return self.__actual_distance[task_id]
    
    # ====================== Actual Pickup Distance Functions
    def add_actual_pickup_distance(self, task_id : int) -> None:
        self.__actual_pickup_distance[task_id] = 0
        
    def update_actual_pickup_distance(self, task_id : int, added_distance : float) -> None:
        self.__actual_pickup_distance[task_id] += added_distance
        
    def remove_actual_pickup_distance(self, task_id : int) -> None:
        del self.__actual_pickup_distance[task_id]
        
    def get_actual_pickup_distance(self, task_id : int) -> float:
        return self.__actual_pickup_distance[task_id]
    
    # ====================== Paths Functions
    
    def add_paths(self, steps : list) -> None:
        for i, step in enumerate(steps):
            if self.__truncated_paths[i] == []:
                self.__truncated_paths[i].append(step)
                self.__paths[i].append(step)
                continue
            self.__paths[i].append(step)
            if self.__truncated_paths[i][-1] != step:
                self.__truncated_paths[i].append(step)
                
    def return_full_paths(self) -> list:
        return self.__paths
    
    def set_soc(self, soc) -> None:
        self.__soc += soc
    
    # ====================== Aisle and Driveway Occupancy Functions
    def add_aisle_occupancy(self, aisle_occupancy : list) -> None:
        """Add occupancy of the aisles to the statistics object per timestep.

        Args:
            aisle_occupancy (list): Occupancy of the aisles per timestep.
        """
        self.__aisle_occupancy.append(aisle_occupancy)
    
    def add_driveway_occupancy(self, driveway_occupancy : list) -> None:
        """Add occupancy of the driveways to the statistics object per timestep.

        Args:
            driveway_occupancy (list): Occupancy of the driveways per timestep.
        """
        self.__driveway_occupancy.append(driveway_occupancy)
    
    # ====================== Completed Task Id Functions
    
    def add_completed_task_id(self, task_id : int, timestep : int, start_location : tuple = None, goal_location : tuple = None, deadline : int = None, sku_id : int = None, inbound_task : bool = None) -> None:
        """Add a completed task with its completion timestep and locations.
        
        Args:
            task_id (int): The ID of the completed task
            timestep (int): The timestep when the task was completed
            start_location (tuple): The start location of the task
            goal_location (tuple): The goal location of the task
        """
        self.__completed_task_ids.append(int(task_id))
        self.__task_completion_timestamps[task_id] = int(timestep)
        if start_location is not None and goal_location is not None:
            self.__completed_task_details[task_id] = (start_location, goal_location, deadline, sku_id, inbound_task)
        
        # Check if task was completed after its deadline
        if task_id in self.__task_deadlines:
            deadline = self.__task_deadlines[task_id]
            if timestep > deadline:
                self.__overdue_task_completions += 1
        
    def get_completed_task_ids(self) -> list:
        return self.__completed_task_ids
            
    # ====================== Completed To-Pickup Task Id Functions
    
    def add_completed_to_pickup_task_id(self, task_id : int) -> None:
        self.__completed_to_pickup_task_ids.append(int(task_id))
        
    # ====================== Runtime Functions
    
    def increment_improved_assignments(self) -> None:
        self.__num_improved_assignments += 1
        
    def increment_worse_assignments(self) -> None:
        self.__num_worse_assignments += 1
        
    def increment_same_assignments(self) -> None:
        self.__num_same_assignments += 1
    
    def set_total_runtime(self, time : float) -> None:
        self.__total_runtime = time
        
    def add_total_CRG_time(self, time : float) -> None:
        self.__CRG_time.append(time)
        
    def add_total_TA_time(self, time : float) -> None:
        self.__TA_time.append(float(time))
        
    def add_total_PF_time(self, time : float) -> None:
        self.__PF_time.append(float(time))
        
    def add_total_SIM_time(self, time : float) -> None:
        self.__SIM_time.append(time)
            

    def add_admisibility(self, admisibility : list) -> None:
        for b in admisibility:
            self.__admisibility.append(b)
    
    def append_admisibility_differences(self, admisibility : list) -> None:
        for b in admisibility:
            self.__admisibility_differences.append(b)
    # ====================== Utils
    def trim_data(self):
        # Trim Actual Distance 
        for task_id in list(self.__actual_distance.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_distance[task_id]
        
        # Trim Estimated Distance 
        for task_id in list(self.__estimated_distance.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_distance[task_id]
                
        # Trim Actual Pickup Distance
        for task_id in list(self.__actual_pickup_distance.copy().keys()):
            if task_id not in self.__completed_to_pickup_task_ids:
                del self.__actual_pickup_distance[task_id]
                
        # Trim Estimated Pickup Distance
        for task_id in list(self.__estimated_pickup_distance.copy().keys()):
            if task_id not in self.__completed_to_pickup_task_ids:
                del self.__estimated_pickup_distance[task_id]
                
        # Trim Actual Duration
        for task_id in list(self.__actual_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_duration[task_id]    
            
        # Trim Estimated Duration
        for task_id in list(self.__estimated_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_duration[task_id]
                
        # Trim Actual Pickup Duration
        for task_id in list(self.__actual_pickup_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__actual_pickup_duration[task_id]
                
        # Trim Estimated Pickup Duration
        for task_id in list(self.__estimated_pickup_duration.copy().keys()):
            if task_id not in self.__completed_task_ids:
                del self.__estimated_pickup_duration[task_id]
            
    def return_throughput(self):
        return (len(self.__completed_task_ids)/self.__T)*60
            
    def return_sum_of_costs(self):
        return np.sum(list(self.__actual_distance.values()))/60

    def compute_stationary_robots(self) -> list:
        num_idle_robots = [int(len(self.__paths))]
        
        # Create numpy array of tuples
        paths = np.asarray(self.__paths, dtype="f,f")
        
        # Save first state
        prev_state = paths.T[0]
        # Iterate over the state of the system at each timestep, checking how many robots are in the same position as the previous state
        for state in paths.T[1:]:
            num_idle_robots.append(int(np.count_nonzero(prev_state == state)))
            prev_state = state
        return num_idle_robots
    
    def compute_velocity_timesteps(self) -> list:
        velocity_timesteps = []
        
        paths = np.asarray(self.__paths, dtype="f,f")
        # Save first state
        prev_state = paths.T[0]
        # Iterate over the state of the system at each timestep, checking how many robots are in the same position as the previous state
        for state in paths.T[1:]:
            velocity = []
            for robot_id in range(len(state)):
                if prev_state[robot_id] == state[robot_id]:
                    velocity.append(0)
                else:
                    velocity.append(1)
            velocity_timesteps.append(velocity)
            prev_state = state  
        return velocity_timesteps
        
    # ====================== Statistical Analysis
    
    def task_length(self) -> list:
        average_task_length = np.average(list(self.__actual_distance.values()))
        average_distance_to_task = np.average(list(self.__actual_pickup_distance.values()))
        average_estimated_task_length = np.average(list(self.__estimated_distance.values()))
        average_estimated_distance_to_task = np.average(list(self.__estimated_pickup_distance.values()))
        
        print("Average Actual Task Distance: ", average_task_length)
        print("Average Actual Distance to Task: ", average_distance_to_task)
        print("Average Estimated Task Distance: ", average_estimated_task_length)
        print("Average Estimated Distance To Task: ", average_estimated_distance_to_task)
            
            
    def print_statistics(self) -> None:
        print("===Runtime Statistics===")
        print("Total Runtime: " + str(self.__total_runtime))
        print("Total Task Generation Time: " + str(np.sum(self.__CRG_time)))
        print("Total Task Allocation Time: " + str(np.sum(self.__TA_time)))
        print("Total Path Planning Time: " + str(np.sum(self.__PF_time)))
        print("Total Simulation Time: " + str(np.sum(self.__SIM_time)))
        
        
        print("===Number of Completed Tasks===")
        print(self.__completed_task_ids)
        
        
        print("===Robot Paths===")
        # TODO: Remove locations in path where robot stays in the same spot for more than one timestep
        for i, robot_path in enumerate(self.__truncated_paths):
            print("Robot " + str(i) + " moved " + str(len(robot_path)) + " spaces during the simulation.")
            
        total = 0
        for robot_path in self.__truncated_paths:
            prev_state = robot_path[0]
            for state in robot_path[1:]:
                if prev_state != state:
                    total += 1
                else:
                    pass
                prev_state = state
        print("Total: ", total)
            
        print("===Estimated Task Distance===")
        
        total = 0
        for distance in list(self.__estimated_distance.values()):
            total += distance
            print(distance)
        print("Estimated Task Distance Total: ", total)
           
           
        print("===Estimated To-Pickup Distance===")
            
        total = 0
        for distance in list(self.__estimated_pickup_distance.values()):
            total += distance
            print(distance)
        print("Estimated Distance To-Pickup Total: ", total)
        
            
        print("===Actual Task Distance===")
        
        total = 0
        for distance in list(self.__actual_distance.values()):
            total += distance
            print(distance)
        print("Actual Task Distance Total: ", total)
            
        total = 0
        print("===Actual To-Pickup Distance===")
            
        for distance in list(self.__actual_pickup_distance.values()):
            total += distance
            print(distance)
        print("Actual Distance To-Pickup Total: ", total)
        
        
        print("===Throughout===")
        print("Throughput: " + str((len(self.__completed_task_ids)/self.__T)*60) + " tasks/min")
        
        print("===Sum of Costs===")
        print("Sum of Costs: " + str(np.sum(list(self.__actual_distance.values()))) + "s or " + str(np.sum(list(self.__actual_distance.values()))/60) + "min.")
        
    def save_data(self, intermediate_output_file = None):
        velocity_timesteps = self.compute_velocity_timesteps()
        
        # print(self.__actual_duration)
        self.remove_uncompleted_task_durations()
        # print(self.__actual_duration)
        
        # Ensure all completed tasks have service times
        for task_id in self.__completed_task_ids:
            if task_id in self.__task_completion_timestamps and task_id in self.__task_release_timestamps:
                if task_id not in self.__service_times:
                    self.update_service_time(task_id, self.__task_completion_timestamps[task_id])
        
        # Calculate final statistics
        avg_service_time, total_service_time = self.get_service_time_stats()
        avg_task_cost, total_costs = self.get_cost_stats()
        
        data = {
            # Input parameters from main
            "task_weight": self.__generated_task_weight,
            "task_weights": self.__task_weights,
            "agent_carried_weight_per_timestep": self.__agent_carried_weights,
            "agent_reserved_weight_per_timestep": self.__agent_reserved_weights,
            "agent_capacity": self.__agent_capacity,
            "agent_loads_per_timestep": self.__agent_loads,
            "agent_carried_task_ids_per_timestep": self.__agent_carried_tasks,
            "seed": self.__seed,
            "num_robots": self.__num_of_robots,
            "time_horizon": self.__T,
            "max_tasks": self.__max_tasks,
            "task_generation_strategy": self.__task_generation_strategy,
            "initial_task_assignment_strategy": self.__initial_task_assignment_strategy,
            "improvement_task_assignment_strategy": self.__improvement_task_assignment_strategy,
            "path_planning_strategy": self.__path_planning_strategy,
            "map_name": self.__map_name,
            "time_limit": self.__time_limit,
            "initial_inventory": self.__initial_inventory,
            "frequency": self.__frequency,
            "inbound_outbound_ratio": self.__inbound_outbound_ratio,
            "num_skus": self.__num_skus,
            "weight_init_method": self.__weight_init_method,
            "cost_calculation_method": self.__cost_calculation_method,
            "removal_operator": self.__removal_operator,
            "repair_operator": self.__repair_operator,
            "acceptance_function": self.__acceptance_function,
            "T_0": self.__T_0,
            "alpha": self.__alpha,
            "deadline_generation_method": self.__deadline_generation_method,
            "deadline_offset": self.__deadline_offset,
            "base_cost_weight": self.__base_cost_weight,
            "deadline_weight": self.__deadline_weight,
            "sku_distribution_weight": self.__sku_distribution_weight,
            "agent_unallocated_penalty": self.__agent_unallocated_penalty,
            # Simulation results
            "timesteps_completed": self.__T,
            "total_completed_tasks": int(len(self.__completed_task_ids)),
            "completed_tasks": self.__completed_task_ids,
            "completed_task_details": self.__completed_task_details,
            "task_completion_timestamps": self.__task_completion_timestamps,
            "task_release_timestamps": self.__task_release_timestamps,
            "service_times": self.__service_times,
            "average_service_time": avg_service_time,
            "total_service_time": total_service_time,
            "average_task_cost": avg_task_cost,
            "sum_of_costs": total_costs,
            "actual_duration_of_task_from_pick_to_place": self.__actual_duration,
            "estimated_duration_of_task_from_pick_to_place": self.__estimated_duration,
            "actual_duration_of_task_from_start_to_pick": self.__actual_pickup_duration,
            "estimated_duration_of_task_from_start_to_pick": self.__estimated_pickup_duration,
            "actual_distance_start_to_pick": self.__actual_pickup_distance,
            "actual_distance_of_task_from_pick_to_place": self.__actual_distance,
            "collisions": int(np.sum(self.__collisions)),
            "stationary_robots": self.compute_stationary_robots(),
            "unallocated_agents": self.__unallocated_agents,
            "throughput (tasks/min)": self.return_throughput(),
            "SoC(min)": self.__soc,
            "Total Runtime": self.__total_runtime,
            "Total Path Planning Runtime": int(np.sum(self.__PF_time)),
            "Opened Nodes": self.__opened_nodes,
            "Expanded Nodes": self.__expanded_nodes,
            "Path Planning Runtimes": self.__PF_time,
            "Total Task Allocation Runtime": int(np.sum(self.__TA_time)),
            "Number of Path Plan Fails": int(self.__num_path_plan_fails),
            "Task Allocation Runtime": self.__TA_time,
            "admissibility": self.__admisibility,
            "admissibility_differences": self.__admisibility_differences,
            "paths": self.__paths,
            "aisle_occupancy": self.__aisle_occupancy,
            "driveway_occupancy": self.__driveway_occupancy,
            "velocity_timesteps": velocity_timesteps,
            "task_reallocations": self.__task_reallocations,
            "agent_statuses_per_timestep": self.__agent_statuses_per_timestep,
            "agent_carrying_skus_per_timestep" : self.__carrying_skus,
            "agent_goal_locations_per_timestep": self.__agent_goal_locations_per_timestep,
            "agent_task_per_timestep": self.__agent_task_per_timestep,
            "tasks_in_system": self.__tasks_in_system,
            "warehouse_full_locations_per_timestep": self.__warehouse_full_locations_per_timestep,
            "warehouse_row_counts_per_timestep": self.__warehouse_row_counts_per_timestep,
            "warehouse_col_counts_per_timestep": self.__warehouse_col_counts_per_timestep,
            "driveway_full_locations_per_timestep": self.__driveway_full_locations_per_timestep,
            "warehouse_sku_counts_per_timestep": self.__warehouse_sku_counts_per_timestep,
            "driveway_sku_counts_per_timestep": self.__driveway_sku_counts_per_timestep,
            "py_lns_logs": self.__py_lns_logs,
            "overdue_task_completions": self.__overdue_task_completions,
            "sku_centroids_per_timestep": self.__sku_centroids_per_timestep,
            "sku_locations_per_timestep": self.__sku_locations_per_timestep,
        }
        
        data.update(self.get_completion_metrics())
        if self.__improvement_task_assignment_strategy == "cbta_dcbs":
            data.update(self.get_cbta_cost_metrics())

        if intermediate_output_file is not None:
            with open(intermediate_output_file, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        else:
            with open(self.__output_file, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

    def add_task_reallocation(self, task_id: int) -> None:
        """Increment the reallocation count for a task."""
        if task_id not in self.__task_reallocations:
            self.__task_reallocations[task_id] = 0
        self.__task_reallocations[task_id] += 1

    def get_task_reallocations(self) -> dict:
        """Get the dictionary of task reallocations.
        
        Returns:
            Dictionary mapping task IDs to their reallocation counts
        """
        return self.__task_reallocations

    def get_task_completion_timestamps(self) -> dict:
        """Get the dictionary of task completion timestamps.
        
        Returns:
            Dictionary mapping task IDs to their completion timesteps
        """
        return self.__task_completion_timestamps

    def add_task_release(self, task_id: int, timestep: int) -> None:
        """Record when a task is released into the system.
        
        Args:
            task_id (int): The ID of the released task
            timestep (int): The timestep when the task was released
        """
        self.__task_release_timestamps[task_id] = timestep

    def update_service_time(self, task_id: int, completion_timestep: int) -> None:
        """Update service time statistics for a completed task.
        
        Args:
            task_id (int): The ID of the completed task
            completion_timestep (int): The timestep when the task was completed
        """
        if task_id in self.__task_release_timestamps:
            service_time = completion_timestep - self.__task_release_timestamps[task_id]
            self.__service_times[task_id] = service_time
            self.__total_service_time += service_time
            self.__average_service_time = self.__total_service_time / len(self.__service_times)

    def update_task_cost(self, task_id: int, duration: float) -> None:
        """Update cost statistics for a task.
        
        Args:
            task_id (int): The ID of the task
            duration (float): The total duration of the task
        """
        self.__total_task_costs[task_id] = duration
        self.__sum_of_costs += duration
        self.__average_task_cost = self.__sum_of_costs / len(self.__total_task_costs)

    def get_service_time_stats(self) -> tuple:
        """Get service time statistics.
        
        Returns:
            tuple: (average_service_time, total_service_time)
        """
        return self.__average_service_time, self.__total_service_time

    def get_cost_stats(self) -> tuple:
        """Get cost statistics.
        
        Returns:
            tuple: (average_task_cost, sum_of_costs)
        """
        return self.__average_task_cost, self.__sum_of_costs

    def add_agent_statuses_and_goals(self, Rs: AgentLoader) -> None:
        """Record agent statuses and goal locations for the current timestep.
        
        Args:
            Rs: AgentLoader containing all agents
        """
        agent_statuses = []
        agent_goal_locations = []
        agent_tasks = []
        
        for agent in Rs.agents:
            agent_statuses.append(agent.status)
            # Append the current task ID for the agent
            if agent.task_sequence:
                agent_tasks.append(agent.task_sequence[0][0])
            else:
                agent_tasks.append(None)

            if agent.status == 1:  # Going to pickup
                goal_location = agent.task_sequence[0][1]  # First task's start location
            elif agent.status == 2:  # Going to delivery
                goal_location = agent.task_sequence[0][2]  # First task's goal location
            else:  # status == 0, idle
                goal_location = None
                
            agent_goal_locations.append(goal_location)
        
        self.__agent_carried_weights.append([a.carried_weight if a.schedule is not None else int(a.get_sku_id_carrying() is not None) for a in Rs.agents])
        self.__agent_reserved_weights.append([a.assigned_weight for a in Rs.agents])
        self.__agent_loads.append([len(a.carried_items) if a.schedule is not None else int(a.get_sku_id_carrying() is not None) for a in Rs.agents])
        self.__agent_carried_tasks.append([list(a.carried_items) if a.schedule is not None else ([a.task_sequence[0][0]] if a.status == 2 else []) for a in Rs.agents])
        self.__agent_statuses_per_timestep.append(agent_statuses)
        self.__agent_goal_locations_per_timestep.append(agent_goal_locations)
        self.__agent_task_per_timestep.append(agent_tasks)

    def append_tasks_in_system(self, num_tasks: int) -> None:
        self.__tasks_in_system.append(num_tasks)

    def append_warehouse_inventory_state(self, warehouse, aisle_locations):
        # Get all full locations
        full_locations = warehouse.get_full_locations()
        self.__warehouse_full_locations_per_timestep.append(len(full_locations))

        # Get row/col counts
        if not aisle_locations:
            self.__warehouse_row_counts_per_timestep.append([])
            self.__warehouse_col_counts_per_timestep.append([])
            return

        # Find bounds
        rows = [loc[0] for loc in aisle_locations]
        cols = [loc[1] for loc in aisle_locations]
        min_row, max_row = min(rows), max(rows)
        min_col, max_col = min(cols), max(cols)

        # Build sets for fast lookup
        full_set = set(full_locations)

        # Row counts
        row_counts = []
        for r in range(min_row, max_row + 1):
            count = sum((r, c) in full_set for c in range(min_col, max_col + 1) if (r, c) in aisle_locations)
            row_counts.append(count)
        self.__warehouse_row_counts_per_timestep.append(row_counts)

        # Col counts
        col_counts = []
        for c in range(min_col, max_col + 1):
            count = sum((r, c) in full_set for r in range(min_row, max_row + 1) if (r, c) in aisle_locations)
            col_counts.append(count)
        self.__warehouse_col_counts_per_timestep.append(col_counts)

    def append_driveway_inventory_state(self, driveway):
        full_locations = driveway.get_full_locations()
        self.__driveway_full_locations_per_timestep.append(len(full_locations))

    def append_sku_inventory_state(self, warehouse, driveway, num_skus):
        warehouse_counts = [len(warehouse.get_sku_instances(sku_id)) for sku_id in range(1, num_skus + 1)]
        driveway_counts = [len(driveway.get_sku_instances(sku_id)) for sku_id in range(1, num_skus + 1)]
        self.__warehouse_sku_counts_per_timestep.append(warehouse_counts)
        self.__driveway_sku_counts_per_timestep.append(driveway_counts)

    def append_sku_centroids(self, warehouse, num_skus):
        """Compute and log the centroid of each SKU (warehouse+driveway) for this timestep."""
        centroids = []
        for sku_id in range(1, num_skus + 1):
            locations = warehouse.get_sku_instances(sku_id)
            if locations:
                arr = np.array(locations)
                centroid = tuple(np.mean(arr, axis=0))
                centroid = tuple(map(float, centroid))
            else:
                centroid = None
            centroids.append(centroid)
        self.__sku_centroids_per_timestep.append(centroids)

    def append_sku_locations(self, warehouse, driveway, num_skus):
        """Log the locations of each SKU for this timestep."""
        locations_per_sku = []
        for sku_id in range(1, num_skus + 1):
            # locations = warehouse.get_sku_instances(sku_id)
            # locations_per_sku.append([tuple(loc) for loc in locations])
            sku_locations = []
            locations = warehouse.get_sku_instances(sku_id)
            for loc in locations:
                sku_locations.append(tuple(loc))
            locations = driveway.get_sku_instances(sku_id)
            for loc in locations:
                sku_locations.append(tuple(loc))
            locations_per_sku.append(sku_locations)
        self.__sku_locations_per_timestep.append(locations_per_sku)

    def append_py_lns_log(self, log: dict):
        self.__py_lns_logs.append(log)

    def add_task_deadline(self, task_id: int, deadline: int) -> None:
        """Record the deadline for a task.
        
        Args:
            task_id (int): The ID of the task
            deadline (int): The deadline timestep for the task
        """
        self.__task_deadlines[task_id] = deadline
    
    def get_overdue_task_completions(self) -> int:
        """Get the number of tasks completed after their deadline.
        
        Returns:
            int: Number of overdue task completions
        """
        return self.__overdue_task_completions
    
    def get_deadline_generation_method(self) -> str:
        """Get the deadline generation method used in the simulation.
        
        Returns:
            str: The deadline generation method
        """
        return self.__deadline_generation_method
