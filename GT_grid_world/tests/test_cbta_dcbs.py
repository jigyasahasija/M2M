import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import Agent, AgentLoader
from src.path_finding_algorithms.dcbs import _path, plan_dcbs
from src.task_allocation import TaskAllocation
from src.task_allocation_algorithms.cbta_dcbs import cbta_dcbs_call, _best_insertion
from src.case_request_generator import get_deadline, generate_task
from src.analysis.statistics import Stats
from src import router
from src.simulate import simulate
from types import SimpleNamespace


class Grid:
    def __init__(self, width=6, height=3):
        self.width, self.height = width, height
        self.warehouse = Mock()
        self.warehouse.get_full_locations.return_value = []
        self.driveway = Mock()
        self.driveway.get_full_locations.return_value = []

    def get_neighbors(self, node, ignore_robots=False):
        x, y = node
        return [(a, b) for a, b in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                if 0 <= a < self.width and 0 <= b < self.height]

    def get_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def query_sku_KD_trees(self, sku, location, count):
        return (np.zeros(count), np.zeros(count))


def task(start, goal, deadline=100):
    return (frozenset([start]), frozenset([goal]), deadline, 1, 0)


class MetricTests(unittest.TestCase):
    def make_stats(self, method="constant", weight=3):
        return Stats(1, 50, "/tmp/unused-metrics.json", "test", "shortest_path",
                     improvement_task_assignment_strategy="cbta_dcbs",
                     deadline_generation_method=method, base_cost_weight=2,
                     deadline_weight=weight)

    def test_realized_cost_and_completion_denominator(self):
        stats = self.make_stats()
        for task_id in (1, 2, 3):
            stats.add_task_release(task_id, 0)
        stats.add_task_deadline(1, 5)
        stats.add_actual_pickup_distance(1)
        stats.update_actual_pickup_distance(1, 2)
        stats.add_actual_distance(1)
        stats.update_actual_distance(1, 4)
        stats.record_assignment_cost(1, 11)
        stats.record_assignment_cost(2, 20)
        stats.add_completed_task_id(1, 8)
        metrics = stats.get_cbta_cost_metrics()
        self.assertEqual(metrics["total_travel_cost"], 12)
        self.assertEqual(metrics["total_deadline_cost"], 9)
        self.assertEqual(metrics["sum_of_costs"], 21)
        self.assertEqual(metrics["average_task_cost"], 21)
        self.assertEqual(metrics["total_estimated_assignment_cost"], 31)
        self.assertEqual(stats.get_completion_metrics()["completion_ratio"], 1 / 3)
        self.assertEqual(stats.get_completion_metrics()["total_uncompleted_tasks"], 2)

    def test_deadline_disabled_and_zero_weight(self):
        for method, weight, expected_lateness in [("none", 3, 0), ("constant", 0, 3)]:
            stats = self.make_stats(method, weight)
            stats.add_task_release(1, 0)
            stats.add_task_deadline(1, 5)
            stats.add_completed_task_id(1, 8)
            metrics = stats.get_cbta_cost_metrics()
            self.assertEqual(metrics["total_deadline_cost"], 0)
            self.assertEqual(metrics["total_completed_lateness_timesteps"], expected_lateness)

    def test_empty_run_and_no_completions(self):
        stats = self.make_stats()
        self.assertIsNone(stats.get_completion_metrics()["completion_ratio"])
        self.assertEqual(stats.get_cbta_cost_metrics()["sum_of_costs"], 0)
        stats.add_task_release(1, 0)
        self.assertEqual(stats.get_completion_metrics()["completion_ratio"], 0)


class AllocationTests(unittest.TestCase):
    def test_deadline_weight_can_be_enabled_and_disabled(self):
        agent = Agent(0, (0, 0))
        entry = (1, (1, 0), (2, 0), 1)
        self.assertEqual(_best_insertion(Grid(), agent, entry, 0, 1, 0)[0], 2)
        self.assertEqual(_best_insertion(Grid(), agent, entry, 0, 1, 3)[0], 5)

    def test_no_deadline_tasks_ignore_positive_deadline_weight(self):
        stats = Mock()
        stats.get_deadline_generation_method.return_value = "none"
        deadline = get_deadline(0, "none", 30)
        tasks = {1: task((1, 0), (2, 0), deadline)}
        agents = AgentLoader([Agent(0, (0, 0))])
        # No-deadline tasks must remain unpenalized beyond the numeric sentinel.
        _, allocations, cost = cbta_dcbs_call(stats, Grid(), agents, tasks,
                                             deadline + 100, 1, 3)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(cost, 2)

    def test_dispatch_and_matching_use_native_agents(self):
        agents = AgentLoader([Agent(7, (0, 0)), Agent(19, (5, 0))])
        tasks = {1: task((0, 1), (0, 2)), 2: task((5, 1), (5, 2))}
        result, allocations, cost = TaskAllocation(
            Mock(), Grid(), agents, tasks, 'fast_greedy', 'cbta_dcbs', '', 0, 'shortest_path')
        self.assertIs(result, agents)
        self.assertEqual([a.task_sequence[0][0] for a in agents.agents], [1, 2])
        self.assertEqual(len(allocations), 2)
        self.assertEqual(cost, 4)

    def test_active_delivery_capacity_and_idempotence(self):
        agent = Agent(7, (0, 0), [(0, (5, 0), (1, 0), 50)], capacity=3)
        agent.status = 2
        agent.sku_id_carrying = 1
        agents, stats = AgentLoader([agent]), Mock()
        tasks = {0: task((5, 0), (1, 0)),
                 1: task((1, 1), (1, 2)), 2: task((2, 1), (2, 2)),
                 3: task((3, 1), (3, 2))}
        cbta_dcbs_call(stats, Grid(), agents, tasks, 0)
        self.assertEqual(agent.task_sequence[0][0], 0)
        self.assertEqual(agent.status, 2)
        self.assertEqual(len(agent.task_sequence), 3)
        before = agent.task_sequence.copy()
        self.assertEqual(cbta_dcbs_call(stats, Grid(), agents, tasks, 1)[1], [])
        self.assertEqual(agent.task_sequence, before)

    def test_shared_inventory_location_is_not_allocated_twice(self):
        agents = AgentLoader([Agent(0, (0, 0)), Agent(1, (5, 0))])
        tasks = {1: task((1, 1), (1, 2)), 2: task((1, 1), (2, 2))}
        cbta_dcbs_call(Mock(), Grid(), agents, tasks, 0)
        self.assertEqual(len(agents.get_all_assigned_tasks()), 1)

    def test_empty_tasks_and_invalid_weights(self):
        agents = AgentLoader([Agent(0, (0, 0))])
        self.assertEqual(cbta_dcbs_call(Mock(), Grid(), agents, {}, 0)[1], [])
        with self.assertRaises(ValueError):
            cbta_dcbs_call(Mock(), Grid(), agents, {}, 0, 0, 0)


class MultiLoadTests(unittest.TestCase):
    def test_inbound_generation_does_not_fill_reserved_delivery(self):
        grid = Grid()
        grid.driveway.get_empty_locations.return_value = [(4, 0)]
        tasks, success, last_id = generate_task('none', 30, [1], [1], 1, 0, 0,
                                               {1: 0}, grid, {}, Mock(),
                                               reserved_locations={(4, 0)})
        self.assertFalse(success)
        self.assertEqual(tasks, {})
        grid.driveway.add_sku_instance.assert_not_called()

    def run_deliveries(self, capacity, weight=1):
        grid = Grid()
        stock = {(1, 0): 1, (2, 0): 1}
        grid.warehouse.get_full_locations.side_effect = lambda: list(stock)
        grid.warehouse.get_empty_locations.return_value = []
        grid.warehouse.get_sku_at_location.side_effect = lambda loc: SimpleNamespace(sku_id=stock[loc])
        grid.warehouse.remove_sku_instance.side_effect = lambda loc: stock.pop(loc)
        grid.warehouse.get_sku_instances.side_effect = lambda sku: [loc for loc, value in stock.items() if value == sku]
        grid.driveway.get_empty_locations.return_value = [(4, 0), (5, 0)]
        grid.set_occupied = Mock()
        grid.update_sku_KD_trees = Mock()
        grid.get_aisle_occupancy = Mock(return_value=[])
        grid.get_driveway_occupancy = Mock(return_value=[])
        stats = Stats(1, 20, '/tmp/unused.json', 'test', 'shortest_path',
                      improvement_task_assignment_strategy='cbta_dcbs',
                      deadline_generation_method='none', agent_capacity=capacity,
                      base_cost_weight=1, deadline_weight=0)
        tasks = {1: task((1, 0), (4, 0)), 2: task((2, 0), (5, 0))}
        tasks = {key: (*value, weight) for key, value in tasks.items()}
        agents = AgentLoader([Agent(0, (0, 0), capacity=capacity)])
        for task_id in tasks:
            stats.add_task_release(task_id, 0)
        loads = []
        for t in range(20):
            cbta_dcbs_call(stats, grid, agents, tasks, t)
            self.assertLessEqual(len(agents.agents[0].task_sequence), capacity)
            self.assertLessEqual(agents.agents[0].assigned_weight, capacity)
            router.pathPlan('', agents, 'dcbs', stats, G=grid)
            simulate(stats, grid, agents, tasks, '', t)
            loads.append(len(agents.agents[0].carried_items))
            self.assertLessEqual(loads[-1], capacity)
            self.assertLessEqual(agents.agents[0].carried_weight, capacity)
            if not tasks:
                break
        self.assertFalse(tasks)
        self.assertFalse(stock)
        self.assertEqual(stats.get_completed_task_ids(), [1, 2])
        self.assertEqual(agents.agents[0].carried_items, {})
        self.assertEqual(agents.agents[0].schedule, [])
        self.assertEqual(agents.agents[0].remaining_capacity, capacity)
        return loads, stats

    def test_fractional_weights_pickup_delivery_and_capacity_release(self):
        loads, _ = self.run_deliveries(3, weight=1.5)
        self.assertEqual(max(loads), 2)
        loads, _ = self.run_deliveries(2, weight=1.5)
        self.assertEqual(max(loads), 1)

    def test_mixed_weights_reserve_capacity_without_splitting(self):
        agents = AgentLoader([Agent(0, (0, 0), capacity=3)])
        tasks = {i: (*task((i, 1), (i, 2)), w) for i, w in [(1, 2), (2, 1), (3, 4)]}
        cbta_dcbs_call(Mock(), Grid(), agents, tasks, 0)
        a = agents.agents[0]
        self.assertEqual({t[0] for t in a.task_sequence}, {1, 2})
        self.assertEqual(a.assigned_weight, 3)
        self.assertEqual(a.carried_weight, 0)
        self.assertEqual(a.remaining_capacity, 0)
        self.assertEqual(cbta_dcbs_call(Mock(), Grid(), agents, tasks, 1)[1], [])

    def test_invalid_task_weights_rejected(self):
        for weight in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                cbta_dcbs_call(Mock(), Grid(), AgentLoader([Agent(0, (0, 0))]),
                               {1: (*task((1, 0), (2, 0)), weight)}, 0)

    def test_two_items_of_same_sku_are_carried_and_delivered(self):
        loads, stats = self.run_deliveries(2)
        self.assertEqual(max(loads), 2)
        # Both items travel three cells while carried, even on shared moves.
        costs = stats.get_cbta_cost_metrics()['completed_task_costs']
        self.assertEqual(costs[1]['travel_distance'], 4)
        self.assertEqual(costs[2]['travel_distance'], 4)

    def test_capacity_one_completes_both_tasks_serially(self):
        loads, _ = self.run_deliveries(1)
        self.assertEqual(max(loads), 1)

    def test_copy_preserves_independent_schedule_and_load(self):
        agent = Agent(0, (0, 0), [(1, (1, 0), (2, 0), 10)], capacity=2)
        agent.enable_schedule()
        agent.set_sku_id_carrying(1)
        clone = AgentLoader([agent]).copy().agents[0]
        clone.carried_items.clear()
        clone.schedule.clear()
        self.assertEqual(agent.carried_items, {1: 1})
        self.assertEqual(len(agent.schedule), 2)
        self.assertEqual(clone.capacity, 2)

    def test_capacity_validation(self):
        for capacity in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                Agent(0, (0, 0), capacity=capacity)


class RoutingTests(unittest.TestCase):
    def test_router_dcbs_replans_after_pickup(self):
        agent = Agent(0, (0, 0), [(1, (1, 0), (2, 0), 100)])
        agent.status = 1
        agents = AgentLoader([agent])
        grid, stats = Grid(), Mock()
        self.assertIs(router.pathPlan("", agents, "dcbs", stats, G=grid), agents)
        self.assertEqual(agent.path_sequence, [(1, 0)])
        agent.state = (1, 0)
        agent.status = 2
        router.pathPlan("", agents, "dcbs", stats, G=grid)
        self.assertEqual(agent.path_sequence, [(2, 0)])

    def test_router_dcbs_requires_graph(self):
        with self.assertRaisesRegex(ValueError, "graph"):
            router.pathPlan("", AgentLoader([]), "dcbs", Mock())

    def test_router_preserves_legacy_planner_interface(self):
        for strategy, module in [("pbs", router.pbs), ("ecbs", router.eecbs)]:
            agent = Agent(0, (0, 0), [(1, (2, 0), (3, 0), 100)])
            agent.status = 1
            agents = AgentLoader([agent])
            def plan(*args):
                self.assertEqual(args, ("test-map", 1, 1, 1.2, [(0, 0)], [(2, 0)]))
                return [[(0, 0), (1, 0), (2, 0)]]
            with patch.object(module, "test_cpp_func", side_effect=plan) as planner:
                router.pathPlan("test-map", agents, strategy, Mock())
                planner.assert_called_once()
            self.assertEqual(agent.path_sequence, [(1, 0), (2, 0)])

    def test_goal_can_be_vacated_for_future_constraint(self):
        route = _path(Grid(), (0, 0), (0, 0), 4, {(2, (0, 0), None)})
        self.assertIsNotNone(route)
        self.assertNotEqual(route[2], (0, 0))

    def test_swapping_agents_make_progress_without_collisions(self):
        agents = AgentLoader([Agent(7, (0, 1)), Agent(19, (2, 1))])
        goals = [(2, 1), (0, 1)]
        for a, goal in zip(agents.agents, goals):
            a.status = 1
            a.task_sequence = [(a.id, goal, goal, 100)]
        stats, grid = Mock(), Grid(3, 3)
        for _ in range(8):
            old = agents.get_agent_states()
            plan_dcbs(grid, agents, stats, horizon=5)
            new = [a.path_sequence[0] for a in agents.agents]
            self.assertEqual(len(set(new)), 2)
            self.assertFalse(old[0] == new[1] and old[1] == new[0])
            for a, position in zip(agents.agents, new):
                a.state = position
        self.assertEqual(agents.get_agent_states(), goals)
        stats.update_num_path_plan_fail.assert_not_called()

    def test_failure_replaces_stale_paths_with_joint_wait(self):
        agents = AgentLoader([Agent(0, (0, 0)), Agent(1, (1, 0))])
        for a, goal in zip(agents.agents, [(1, 0), (0, 0)]):
            a.status = 1
            a.task_sequence = [(a.id, goal, goal, 100)]
            a.path_sequence = [goal]
        stats = Mock()
        plan_dcbs(Grid(2, 1), agents, stats, max_expansions=1)
        self.assertEqual([a.path_sequence for a in agents.agents], [[(0, 0)], [(1, 0)]])
        stats.update_num_path_plan_fail.assert_called_once()


if __name__ == '__main__':
    unittest.main()
