from typing import Tuple, Dict

from .analysis.statistics import Stats
from .graph import Graph
from .agent import *
from .utils import *


def _refresh_tasks_after_warehouse_change(J : set, G : Graph, changed_task_id : int, sku_id : int) -> None:
    """
    Ensure all tasks referencing warehouse locations remain consistent with the
    current inventory layout after a SKU is removed or added.
    """
    if not J:
        return

    for other_task_id in list(J.keys()):
        if other_task_id == changed_task_id:
            continue

        task_tuple = J[other_task_id]
        start_locations, goal_locations, deadline, task_sku_id, task_type = task_tuple[:5]

        if task_sku_id == sku_id:
            if task_type == 0:
                new_start_locs = frozenset(G.warehouse.get_sku_instances(sku_id))
                new_goal_locs = frozenset(G.driveway.get_empty_locations())
                J[other_task_id] = (new_start_locs, new_goal_locs, deadline, task_sku_id, task_type) + task_tuple[5:]
                continue
            elif task_type == 1:
                new_goal_locs = frozenset(G.warehouse.get_empty_locations())
                J[other_task_id] = (start_locations, new_goal_locs, deadline, task_sku_id, task_type) + task_tuple[5:]
                continue
        else:
            if task_type == 0:
                new_goal_locs = frozenset(G.driveway.get_empty_locations())
                J[other_task_id] = (start_locations, new_goal_locs, deadline, task_sku_id, task_type) + task_tuple[5:]
                continue
            elif task_type == 1:
                new_goal_locs = frozenset(G.warehouse.get_empty_locations())
                J[other_task_id] = (start_locations, new_goal_locs, deadline, task_sku_id, task_type) + task_tuple[5:]
                continue


def simulate(S : Stats, G : Graph, Rs : AgentLoader, J : Dict[int, Tuple], map_name : str, t : int) -> Tuple[AgentLoader, set]:
    """
    Simulate the system for one timestep.

    Args:
        S (Stats): Statistics object
        G (Graph): Graph object
        Rs (AgentLoader): AgentLoader object
        J (Dict[int, Tuple]): Dictionary of tasks
        map_name (str): Name of the map
        t (int): Current timestep

    Returns:
        Tuple[AgentLoader, Dict[int, Tuple]]: Updated AgentLoader object and updated dictionary of tasks
    """

    # Update state of robots
    for agent in Rs.agents:
        # If robot sequence is stationary, leave the robot in place (wait action)
        if len(agent.path_sequence) == 0:
            continue
        # If robot does have a sequence of actions, pop next state and update
        else:
            # Update Agent State and Graph Occupied States
            G.set_occupied(agent.state, False)
            old_state = agent.state
            agent.state = agent.path_sequence.pop(0)
            G.set_occupied(agent.state, True)
            
            if agent.schedule is not None:
                # Each carried item's travel includes detours for other events.
                for task_id in agent.carried_items:
                    if old_state != agent.state:
                        S.update_actual_distance(task_id, 1)
                    S.update_actual_duration(task_id, S.get_actual_duration(task_id) + 1)
                if agent.status == 1:
                    task_id = agent.task_sequence[0][0]
                    if old_state != agent.state:
                        S.update_actual_pickup_distance(task_id, 1)
                    S.update_actual_pickup_duration(task_id, S.get_actual_pickup_duration(task_id) + 1)
            elif agent.status == 1:
                if old_state != agent.state:
                    S.update_actual_pickup_distance(agent.task_sequence[0][0], 1)
                S.update_actual_pickup_duration(agent.task_sequence[0][0], S.get_actual_pickup_duration(agent.task_sequence[0][0]) + 1)
                
            elif agent.status == 2:
                if agent.get_sku_id_carrying() is None:
                    raise ValueError(f"Agent {agent.id} is not carrying any item when it is in the delivery phase for task {agent.task_sequence[0][0]}: agent status {agent.status}")
                if old_state != agent.state:
                    S.update_actual_distance(agent.task_sequence[0][0], 1)
                S.update_actual_duration(agent.task_sequence[0][0], S.get_actual_duration(agent.task_sequence[0][0]) + 1)
            else:
                pass
    
    S.append_number_of_collisions(Rs.detect_collisions())
    
    # Update Statistics for the total paths taken
    S.add_paths(Rs.get_agent_states())
    
    # Update Statistics for Asile and Driveway Occupancy
    S.add_aisle_occupancy(G.get_aisle_occupancy())
    S.add_driveway_occupancy(G.get_driveway_occupancy())
    
    for agent in Rs.agents:
        if agent.status == 1:
            if agent.state == agent.task_sequence[0][1]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                deadline = task[3]

                if agent.schedule is not None:
                    if agent.carried_weight + task_weight(agent.task_sequence[0]) > agent.capacity + 1e-9:
                        raise ValueError("Pickup would exceed robot capacity")
                    inventory = (G.warehouse if start_location in G.warehouse.get_full_locations()
                                 else G.driveway)
                    if start_location not in inventory.get_full_locations():
                        raise ValueError(f"No inventory at pickup {start_location}")
                    if inventory.get_sku_at_location(start_location).sku_id != J[task_id][3]:
                        raise ValueError(f"Wrong SKU at pickup for task {task_id}")
                
                # Outbound or warehouse-based pickup
                if start_location in G.warehouse.get_full_locations():
                    try:
                        sku_id = G.warehouse.get_sku_at_location(start_location).sku_id
                        if sku_id is None:
                            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
                        agent.set_sku_id_carrying(G.warehouse.get_sku_at_location(start_location).sku_id)
                        G.warehouse.remove_sku_instance(start_location)
                        G.update_sku_KD_trees(agent.get_sku_id_carrying())

                        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)

                        if agent.get_sku_id_carrying() is None:
                            raise ValueError(f"Agent should be holding item after pickup ... Exiting")
                        
                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from warehouse at {start_location}: {e}")
                        exit()
                # Inbound: picking up from driveway (now empty)
                elif start_location in G.driveway.get_full_locations():
                    try:
                        # Only save sku id
                        print(f"Agent {agent.id} picking up task {task_id} with sku {G.driveway.get_sku_at_location(start_location)}")
                        sku_id = G.driveway.get_sku_at_location(start_location).sku_id
                        if sku_id is None:
                            raise ValueError(f"No item for agent {agent.id} at {start_location} found ... Exiting")
                        agent.set_sku_id_carrying(sku_id)
                        G.driveway.remove_sku_instance(start_location)
                        
                        _refresh_tasks_after_warehouse_change(J, G, task_id, sku_id)

                        if agent.get_sku_id_carrying() is None:
                            raise ValueError(f"Agent should be holding item after pickup ... Exiting")

                    except Exception as e:
                        print(f"[WARN] Could not remove SKU from driveway at {start_location}: {e}")

                S.add_completed_to_pickup_task_id(task_id)
                if agent.schedule is not None:
                    agent.schedule.pop(0)
                    agent.sync_schedule()
                else:
                    agent.status = 2
        elif agent.status == 2:
            if agent.state == agent.task_sequence[0][2]:
                task = agent.task_sequence[0]
                task_id = task[0]
                start_location = task[1]
                goal_location = task[2]
                deadline = task[3]
                
                deadline = J[task_id][2]
                sku_id = J[task_id][3]
                inbound_task = J[task_id][4]

                if sku_id != agent.get_sku_id_carrying():
                    raise ValueError(f"Agent {agent.id} carrying sku {agent.get_sku_id_carrying()} but task {task_id} requires sku {sku_id} ... Exiting")

                if agent.schedule is not None and goal_location not in (set(G.warehouse.get_empty_locations()) | set(G.driveway.get_empty_locations())):
                    raise ValueError(f"Delivery location {goal_location} is occupied")

                # Inbound task: dropping off to warehouse
                if goal_location in G.warehouse.get_empty_locations():
                    print(f"Agent {agent.id} dropping off task {task_id} with sku {agent.get_sku_id_carrying()}")
                    carried_sku = agent.get_sku_id_carrying()
                    G.warehouse.add_sku_instance(carried_sku, goal_location)
                    G.update_sku_KD_trees(agent.get_sku_id_carrying())
                    _refresh_tasks_after_warehouse_change(J, G, task_id, carried_sku)

                                
                elif goal_location in G.driveway.get_empty_locations():
                    pass
                agent.set_sku_id_carrying(None)

                S.add_completed_task_id(task_id, t, start_location, goal_location, int(deadline), int(sku_id), int(inbound_task))
                S.update_service_time(task_id, t)
                
                J.pop(task_id)
                    
                agent.task_sequence.pop(0)
                if agent.schedule is not None:
                    agent.schedule.pop(0)
                    agent.sync_schedule()
                    continue
                if agent.task_sequence == []:
                    agent.status = 0
                else:
                    agent.status = 1
                    new_task_id = agent.task_sequence[0][0]
                    S.add_actual_distance(new_task_id)
                    S.add_actual_pickup_distance(new_task_id)
                    
                    # Initialize durations for new task 
                    S.add_actual_duration(new_task_id)
                    S.add_actual_pickup_duration(new_task_id)

                    estimated_to_pickup_path = G.get_distance(agent.state, agent.task_sequence[0][1])
                    estimated_task_path = G.get_distance(agent.task_sequence[0][1], agent.task_sequence[0][2])

                    S.add_estimated_pickup_duration(new_task_id, estimated_to_pickup_path)
                    S.add_estimated_pickup_distance(new_task_id, estimated_to_pickup_path)

                    S.add_estimated_distance(new_task_id, estimated_task_path)
                    S.add_estimated_duration(new_task_id, estimated_task_path)
                    
                    if t <= 100:
                        S.append_early_task_ids(new_task_id)
                        
    S.append_carrying_skus(Rs.get_all_agent_carrying_skus())
    return Rs, J
