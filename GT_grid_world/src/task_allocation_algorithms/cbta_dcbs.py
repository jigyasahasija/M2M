"""Cost-based assignment for weighted multi-load pickup and delivery.

Pickup and delivery events can interleave. Each assigned task reserves its
weight until delivery. Binary optimization handles indivisible weighted tasks;
the paper's unit-weight flow solver and packaging are separate work.
"""

import math

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import lil_matrix
from ..agent import task_weight

from .initial_solutions.construct_cost_elements import construct_cost_elements


def _best_insertion(G, agent, task, t, distance_weight, delay_weight):
    agent.enable_schedule()
    if task_weight(task) > agent.remaining_capacity + 1e-9:
        return None
    tasks = {entry[0]: entry for entry in agent.task_sequence}
    tasks[task[0]] = task
    best = None
    # Insert pickup and delivery independently, retaining all existing events.
    for pickup in range(len(agent.schedule) + 1):
        events = agent.schedule.copy()
        events.insert(pickup, (task[0], 1))
        for delivery in range(pickup + 1, len(events) + 1):
            candidate = events.copy()
            candidate.insert(delivery, (task[0], 2))
            carried = set(agent.carried_items)
            location, distance, completion_distance = agent.state, 0, None
            valid = True
            for task_id, phase in candidate:
                if phase == 1:
                    if task_id in carried or sum(task_weight(tasks[i]) for i in carried) + task_weight(tasks[task_id]) > agent.capacity + 1e-9:
                        valid = False
                        break
                    carried.add(task_id)
                else:
                    if task_id not in carried:
                        valid = False
                        break
                    carried.remove(task_id)
                destination = tasks[task_id][phase]
                distance += G.get_distance(location, destination)
                location = destination
                if task_id == task[0] and phase == 2:
                    completion_distance = distance
            if not valid or carried or not math.isfinite(distance):
                continue
            cost = distance_weight * completion_distance
            if delay_weight > 0:
                cost += delay_weight * max(0, t + completion_distance - task[3])
            # Break equal-cost ties using total schedule travel.
            result = (cost, distance, pickup, delivery)
            if best is None or result < best:
                best = result
    return (best[0], best[2], best[3]) if best is not None else None


def cbta_dcbs_call(S, G, Rs, J, t, base_cost_weight=1.0, deadline_weight=0.0):
    """Return the usual (agents, allocations, cost) allocation result.

Reuse main's inventory masks and cached graph distances. Recompute candidates
after matching when tasks compete for the same physical inventory locations.
DCBS routing is selected by the normal simulator routing stage.
"""
    if (not math.isfinite(base_cost_weight) or not math.isfinite(deadline_weight)
            or min(base_cost_weight, deadline_weight) < 0
            or base_cost_weight + deadline_weight == 0):
        raise ValueError("CBTA cost weights must be finite, nonnegative, and not both zero")
    # Keep main's numeric no-deadline representation, but never penalize it,
    # even if the user left a positive weight configured for another run.
    if S.get_deadline_generation_method() == "none":
        deadline_weight = 0.0
    for task_id, entry in J.items():
        weight = task_weight((task_id, None, None, entry[2], entry[5] if len(entry) > 5 else 1.0))
        S.record_task_weight(task_id, weight)
    for agent in Rs.agents:
        agent.enable_schedule()
    allocations, total_cost = [], 0.0
    while True:
        assigned = {task[0] for task in Rs.get_all_assigned_tasks()}
        if not (J.keys() - assigned):
            break
        available_agents = [m for m, agent in enumerate(Rs.agents)
                 if agent.remaining_capacity > 1e-9]
        if not available_agents:
            break
        elements = construct_cost_elements(J, Rs, G, t, "shortest_path")
        _, _, start_mask, goal_mask, starts, goals, task_ids, *_ = elements
        candidates = {}
        for m in available_agents:
            for n, task_id in task_ids.items():
                for p in np.flatnonzero(start_mask[n]):
                    for q in np.flatnonzero(goal_mask[n]):
                        task = (task_id, starts[p], goals[q], J[task_id][2], J[task_id][5] if len(J[task_id]) > 5 else 1.0)
                        insertion = _best_insertion(G, Rs.agents[m], task, t,
                                                    base_cost_weight, deadline_weight)
                        if insertion is None:
                            continue
                        candidate = (*insertion, int(p), int(q))
                        if (m, n) not in candidates or candidate < candidates[m, n]:
                            candidates[m, n] = candidate
        if not candidates:
            break
        # Unsplittable weighted tasks require binary assignment variables.
        # First maximize task count, then minimize cost at that cardinality.
        pairs = sorted(candidates)
        matrix = lil_matrix((len(Rs.agents) + len(task_ids), len(pairs)))
        for column, (m, n) in enumerate(pairs):
            entry = J[task_ids[n]]
            matrix[m, column] = entry[5] if len(entry) > 5 else 1.0
            matrix[len(Rs.agents) + n, column] = 1
        limits = [a.remaining_capacity for a in Rs.agents] + [1] * len(task_ids)
        constraint = LinearConstraint(matrix.tocsr(), 0, limits)
        options = dict(integrality=np.ones(len(pairs)), bounds=Bounds(0, 1))
        maximum = milp(-np.ones(len(pairs)), constraints=constraint, **options)
        if not maximum.success:
            raise RuntimeError(f"Weighted assignment failed: {maximum.message}")
        cardinality = int(round(sum(maximum.x)))
        minimum = milp([candidates[pair][0] for pair in pairs],
                       constraints=[constraint, LinearConstraint(np.ones((1, len(pairs))), cardinality, cardinality)],
                       **options)
        if not minimum.success:
            raise RuntimeError(f"Weighted assignment failed: {minimum.message}")
        selected = [pair for pair, value in zip(pairs, minimum.x) if value > 0.5]
        used_locations = set()
        committed = 0
        for m, n in selected:
            task_id = task_ids[n]
            _, _, _, p, q = candidates[m, n]
            if starts[p] in used_locations or goals[q] in used_locations:
                continue
            task = (task_id, starts[p], goals[q], J[task_id][2], J[task_id][5] if len(J[task_id]) > 5 else 1.0)
            cost, pickup, delivery = _best_insertion(G, Rs.agents[m], task, t,
                                         base_cost_weight, deadline_weight)
            Rs.agents[m].task_sequence.append(task)
            Rs.agents[m].schedule.insert(pickup, (task_id, 1))
            Rs.agents[m].schedule.insert(delivery, (task_id, 2))
            Rs.agents[m].sync_schedule()
            for update in (S.append_early_task_ids, S.add_actual_distance,
                           S.add_actual_pickup_distance, S.add_actual_duration,
                           S.add_actual_pickup_duration):
                update(task_id)
            used_locations.update((starts[p], goals[q]))
            allocations.append((m, task_id, p, q))
            S.record_assignment_cost(task_id, cost)
            S.record_task_weight(task_id, task_weight(task))
            total_cost += cost
            committed += 1
        if not committed:
            break
    return Rs, allocations, total_cost
