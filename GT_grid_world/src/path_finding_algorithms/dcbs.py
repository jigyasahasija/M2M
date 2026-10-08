"""Finite-horizon conflict-based routing for the existing simulator.

Plan to each agent's next pickup/delivery and execute one step before replanning.
Constraints include waiting agents, vertex collisions, and opposite edge moves.
"""

from heapq import heappop, heappush
from itertools import combinations, count

def _path(G, start, goal, horizon, constraints):
    """Space-time A* with waiting and a closest-endpoint horizon fallback.

Unlike stopping at the first goal visit, searching through the horizon checks
future constraints at the goal and permits leaving it when necessary.
"""
    order = count()
    queue = [(G.get_distance(start, goal), 0, next(order), start, 0)]
    parents = {(start, 0): None}
    while queue:
        _, _, _, location, time = heappop(queue)
        if time == horizon:
            route, state = [], (location, time)
            while state is not None:
                route.append(state[0])
                state = parents[state]
            return route[::-1]
        # Prefer holding position once the next simulator stop is reached.
        neighbors = G.get_neighbors(location, True)
        moves = [location, *neighbors] if location == goal else [*neighbors, location]
        for target in moves:
            state = target, time + 1
            if (time + 1, target, None) in constraints or (time, location, target) in constraints:
                continue
            if state in parents:
                continue
            parents[state] = location, time
            h = G.get_distance(target, goal)
            heappush(queue, (time + 1 + h, h, next(order), target, time + 1))
    return None


def _conflicts(paths):
    conflicts = []
    for time in range(1, len(paths[0])):
        for a, b in combinations(range(len(paths)), 2):
            if paths[a][time] == paths[b][time]:
                conflicts.append((a, b, (time, paths[a][time], None),
                                   (time, paths[b][time], None)))
            elif (paths[a][time - 1] == paths[b][time]
                  and paths[b][time - 1] == paths[a][time]):
                conflicts.append((a, b, (time - 1, paths[a][time - 1], paths[a][time]),
                                   (time - 1, paths[b][time - 1], paths[b][time])))
    return conflicts


def plan_dcbs(G, Rs, S, horizon=10, max_expansions=1000):
    """Set one collision-free step for every agent, or a safe joint wait.

The next call sees pickup/delivery transitions performed by simulate.py. No
separate world state or task lifecycle is maintained by this planner.
"""
    if horizon < 1 or max_expansions < 1:
        raise ValueError("DCBS horizon and expansion limit must be positive")
    if not Rs.agents:
        return Rs
    starts = Rs.get_agent_states()
    if len(starts) != len(set(starts)):
        raise ValueError("DCBS requires distinct starting positions")
    goals = [a.task_sequence[0][a.status] if a.status else a.state for a in Rs.agents]
    root = [_path(G, start, goal, horizon, frozenset()) for start, goal in zip(starts, goals)]
    order, queue, visited = count(), [], set()

    def enqueue(paths, constraints):
        if any(path is None for path in paths):
            return
        conflicts = _conflicts(paths)
        # Remaining distance keeps partial paths moving toward their goals;
        # time away from the goal penalizes unnecessary detours and waits.
        cost = sum(sum(location != goals[i] for location in path[1:])
                   + G.get_distance(path[-1], goals[i])
                   for i, path in enumerate(paths))
        heappush(queue, (cost + len(conflicts), next(order), paths, constraints, conflicts))

    enqueue(root, frozenset())
    for _ in range(max_expansions):
        if not queue:
            break
        _, _, paths, constraints, conflicts = heappop(queue)
        if not conflicts:
            for agent, path in zip(Rs.agents, paths):
                agent.path_sequence = [path[1]]
            return Rs
        a, b, first, second = conflicts[0]
        for index, restriction in ((a, first), (b, second)):
            child_constraints = constraints | {(index, restriction)}
            if child_constraints in visited:
                continue
            visited.add(child_constraints)
            child = paths.copy()
            child[index] = _path(G, starts[index], goals[index], horizon,
                                 {c for i, c in child_constraints if i == index})
            enqueue(child, child_constraints)
    S.update_num_path_plan_fail()
    for agent in Rs.agents:
        agent.path_sequence = [agent.state]
    return Rs
