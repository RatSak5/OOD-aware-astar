"""Faithful SeeA* sampler, search, and matched-benchmark tests."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import numpy as np

from algorithms.movement import get_neighbors, get_transition_cost
from data.dataset import make_dataset_sample
from data.map_generator import MapGenerationConfig
from seeastar.benchmark import run_search_benchmarks
from seeastar.original import SeeAStarConfig, seeastar
from seeastar.sampling import (
    CompetitiveClusterSampler,
    UCTLikeSampler,
    UniformSampler,
    lowest_f_candidate,
)
from seeastar.uniform import UniformSeeAStarConfig, uniform_seeastar


def _sample(distribution: str, seed: int):
    return make_dataset_sample(
        MapGenerationConfig(height=12, width=12, distribution=distribution, seed=seed)
    )


class CandidateSamplingTests(unittest.TestCase):
    def test_uniform_sampling_is_unique_bounded_and_reproducible(self) -> None:
        open_nodes = [(0, col) for col in range(12)]
        f_values = {node: float(node[1]) for node in open_nodes}
        depths = {node: node[1] for node in open_nodes}
        first = UniformSampler(k=4, seed=991).select(open_nodes, f_values, depths)
        second = UniformSampler(k=4, seed=991).select(open_nodes, f_values, depths)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertEqual(len(set(first)), 4)
        self.assertTrue(set(first).issubset(open_nodes))

    def test_uniform_sampler_has_balanced_candidate_inclusion_frequencies(self) -> None:
        """Check uniformity empirically over independent fixed-seed trials."""

        open_nodes = [(0, col) for col in range(5)]
        f_values = {node: float(node[1]) for node in open_nodes}
        depths = {node: node[1] for node in open_nodes}
        inclusion_counts = {node: 0 for node in open_nodes}
        trials = 2000
        for seed in range(trials):
            selected = UniformSampler(k=2, seed=seed).select(
                open_nodes, f_values, depths
            )
            self.assertEqual(len(selected), 2)
            self.assertEqual(len(set(selected)), 2)
            for node in selected:
                inclusion_counts[node] += 1
        expected = trials * 2 / len(open_nodes)
        self.assertTrue(
            all(abs(count - expected) < 75 for count in inclusion_counts.values()),
            inclusion_counts,
        )

    def test_expansion_ranking_selects_minimum_f_from_sampled_candidates(self) -> None:
        open_nodes = [(0, col) for col in range(8)]
        f_values = {node: float(10 - node[1]) for node in open_nodes}
        depths = {node: node[1] for node in open_nodes}
        candidates = UniformSampler(k=3, seed=41).select(
            open_nodes, f_values, depths
        )
        self.assertEqual(
            lowest_f_candidate(candidates, f_values),
            min(candidates, key=lambda node: (f_values[node], node)),
        )
        self.assertNotEqual(lowest_f_candidate(candidates, f_values), open_nodes[-1])

    def test_lowest_f_candidate_handles_ties_and_empty_sets(self) -> None:
        candidates = [(1, 0), (0, 1)]
        self.assertEqual(
            lowest_f_candidate(candidates, {node: 2.0 for node in candidates}),
            (0, 1),
        )
        with self.assertRaises(ValueError):
            lowest_f_candidate([], {})

    def test_uniform_api_cannot_select_cluster_or_uct(self) -> None:
        config_fields = {field.name for field in fields(UniformSeeAStarConfig)}
        self.assertTrue({"k", "seed", "max_expansions"}.issubset(config_fields))
        self.assertTrue(
            {"sampling_strategy", "num_clusters", "eta", "cb"}.isdisjoint(
                config_fields
            )
        )

        result = uniform_seeastar(
            np.zeros((2, 2), dtype=np.uint8),
            (0, 0),
            (1, 1),
            config=UniformSeeAStarConfig(k=10, seed=7),
        )
        self.assertEqual(result.sampling_strategy, "uniform")
        self.assertEqual(result.candidate_sizes, (1, 3))

    def test_uniform_and_uct_return_all_small_open_sets(self) -> None:
        open_nodes = [(0, 0), (0, 1)]
        f_values = {node: float(index) for index, node in enumerate(open_nodes)}
        depths = {node: index for index, node in enumerate(open_nodes)}
        for sampler in (UniformSampler(5, seed=1), UCTLikeSampler(5, cb=0.35)):
            with self.subTest(sampler=sampler.name):
                self.assertEqual(sampler.select(open_nodes, f_values, depths), open_nodes)

    def test_uct_uses_depth_adjustment_for_candidates_not_final_f_ranking(self) -> None:
        open_nodes = [(0, 0), (0, 1), (0, 2), (0, 3)]
        f_values = {
            (0, 0): 3.5,
            (0, 1): 3.0,
            (0, 2): 7.0,
            (0, 3): 8.0,
        }
        depths = {
            (0, 0): 0,
            (0, 1): 10,
            (0, 2): 0,
            (0, 3): 0,
        }
        sampler = UCTLikeSampler(k=2, cb=1.0)
        candidates = sampler.select(open_nodes, f_values, depths)
        # Candidate membership uses E, but expansion ranks candidates by f,
        # so the deep (0,1) wins despite (0,0) having the lower E score.
        self.assertEqual(set(candidates), {(0, 0), (0, 1)})
        self.assertEqual(min(candidates, key=lambda node: (f_values[node], node)), (0, 1))

    def test_cluster_membership_tracks_open_set_without_duplicates(self) -> None:
        sampler = CompetitiveClusterSampler(
            k=3, shape=(8, 8), num_clusters=2, eta=0.2, seed=31
        )
        open_nodes = [(0, 0), (0, 1), (0, 2), (6, 6), (7, 7)]
        for node in open_nodes:
            sampler.on_add(node)
        sampler.assert_open_consistent(open_nodes)
        memberships = sampler.cluster_members
        flattened = set().union(*memberships)
        self.assertEqual(flattened, set(open_nodes))
        self.assertEqual(sum(len(group) for group in memberships), len(open_nodes))
        chosen = sampler.select(open_nodes, {}, {})
        self.assertEqual(len(chosen), len(set(chosen)))
        self.assertTrue(set(chosen).issubset(open_nodes))
        self.assertLessEqual(len(chosen), 4)  # Nc * ceil(K/Nc)

        sampler.on_remove(open_nodes[0])
        open_nodes.remove((0, 0))
        sampler.assert_open_consistent(open_nodes)
        sampler.on_add((4, 4))
        open_nodes.append((4, 4))
        sampler.assert_open_consistent(open_nodes)

    def test_cluster_returns_all_when_open_is_smaller_than_candidate_budget(self) -> None:
        sampler = CompetitiveClusterSampler(
            k=8, shape=(8, 8), num_clusters=5, eta=0.15, seed=32
        )
        open_nodes = [(2, 3), (4, 5)]
        for node in open_nodes:
            sampler.on_add(node)
        self.assertEqual(sampler.select(open_nodes, {}, {}), open_nodes)


class SeeAStarSearchTests(unittest.TestCase):
    def test_all_original_strategies_find_valid_paths_with_shared_unit_costs(self) -> None:
        grid = np.zeros((7, 9), dtype=np.uint8)
        grid[2, 1:7] = 1
        grid[2, 4] = 0
        start, goal = (0, 0), (6, 8)
        for strategy in ("uniform", "cluster", "uct"):
            with self.subTest(strategy=strategy):
                result = seeastar(
                    grid,
                    start,
                    goal,
                    config=SeeAStarConfig(
                        k=4,
                        sampling_strategy=strategy,
                        num_clusters=2,
                        seed=81,
                    ),
                )
                self.assertTrue(result.found)
                self.assertEqual(result.path[0], start)
                self.assertEqual(result.path[-1], goal)
                measured = 0.0
                for source, destination in zip(result.path, result.path[1:]):
                    self.assertIn(destination, get_neighbors(grid, source))
                    measured += get_transition_cost(grid, source, destination)
                self.assertAlmostEqual(measured, result.found_ground_truth_cost)
                self.assertGreaterEqual(result.generated_nodes, len(result.path))
                self.assertLessEqual(result.generated_nodes, int(np.count_nonzero(grid == 0)))
                self.assertEqual(len(result.expanded_order), len(set(result.expanded_order)))
                self.assertGreaterEqual(result.max_open_size, 1)
                self.assertGreaterEqual(result.final_open_size, 0)
                self.assertGreaterEqual(result.sampling_time_seconds, 0.0)
                self.assertGreaterEqual(
                    result.search_time_seconds, result.sampling_time_seconds
                )

    def test_search_is_reproducible_for_fixed_seed(self) -> None:
        grid = np.zeros((8, 9), dtype=np.uint8)
        grid[2:6, 4] = 1
        grid[4, 4] = 0
        config = SeeAStarConfig(
            k=3,
            sampling_strategy="uniform",
            seed=12345,
        )
        first = seeastar(grid, (0, 0), (7, 8), config=config)
        second = seeastar(grid, (0, 0), (7, 8), config=config)
        self.assertEqual(first.path, second.path)
        self.assertEqual(first.expanded_order, second.expanded_order)
        self.assertEqual(first.candidate_sizes, second.candidate_sizes)
        self.assertEqual(first.found_ground_truth_cost, second.found_ground_truth_cost)

    def test_graph_states_are_merged_without_duplicate_open_entries(self) -> None:
        grid = np.zeros((8, 9), dtype=np.uint8)
        result = uniform_seeastar(
            grid,
            (0, 0),
            (7, 8),
            config=UniformSeeAStarConfig(k=100, seed=9),
        )
        self.assertTrue(result.found)
        self.assertLessEqual(result.generated_nodes, int(np.count_nonzero(grid == 0)))
        self.assertEqual(len(result.expanded_order), len(set(result.expanded_order)))
        self.assertGreaterEqual(result.max_open_size, result.final_open_size)

    def test_goal_and_expansion_limit_termination(self) -> None:
        same = seeastar(
            np.zeros((2, 2), dtype=np.uint8),
            (1, 1),
            (1, 1),
            config=SeeAStarConfig(k=1),
        )
        self.assertTrue(same.found)
        self.assertEqual(same.path, ((1, 1),))
        self.assertEqual(same.found_ground_truth_cost, 0.0)
        self.assertEqual(same.expanded_nodes, 0)
        self.assertEqual(same.final_open_size, 0)
        self.assertEqual(same.mean_open_size, 1.0)

        limited = seeastar(
            np.zeros((8, 8), dtype=np.uint8),
            (0, 0),
            (7, 7),
            config=SeeAStarConfig(k=1, max_expansions=1, seed=1),
        )
        self.assertFalse(limited.found)
        self.assertEqual(limited.stop_reason, "max_expansions")
        self.assertEqual(limited.expanded_nodes, 1)
        self.assertGreater(limited.final_open_size, 0)

    def test_unreachable_and_corner_cutting_cases(self) -> None:
        blocked = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        unreachable = seeastar(blocked, (0, 0), (1, 1))
        self.assertFalse(unreachable.found)
        self.assertIsNone(unreachable.path)
        self.assertEqual(unreachable.stop_reason, "open_exhausted")


class SeeAStarBenchmarkTests(unittest.TestCase):
    def test_matched_benchmark_reports_each_paper_sampler_by_distribution(self) -> None:
        samples = [
            _sample(distribution, 17000 + index)
            for index, distribution in enumerate(("D1", "D2", "D3", "D4", "D5", "D6"))
        ]
        with tempfile.TemporaryDirectory() as directory:
            results = run_search_benchmarks(
                samples,
                directory,
                seeastar_config=SeeAStarConfig(
                    k=8,
                    num_clusters=3,
                    eta=0.15,
                    cb=0.35,
                    seed=100,
                ),
            )
            self.assertEqual(len(results["per_map"]), 6 * 5)
            self.assertEqual(
                {row["method"] for row in results["per_map"]},
                {
                    "Dijkstra",
                    "Conventional A*",
                    "SeeA* (Uniform)",
                    "SeeA* (Cluster)",
                    "SeeA* (UCT)",
                },
            )
            self.assertTrue(
                (Path(directory) / "search_benchmark_per_map.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "search_benchmark_summary.csv").is_file()
            )
            self.assertTrue(
                (Path(directory) / "search_benchmark_config.json").is_file()
            )
            for method in (
                "Dijkstra",
                "Conventional A*",
                "SeeA* (Uniform)",
                "SeeA* (Cluster)",
                "SeeA* (UCT)",
            ):
                overall = next(
                    row
                    for row in results["summary"]
                    if row["method"] == method and row["distribution"] == "all"
                )
                self.assertEqual(overall["success_rate"], 1.0)
            for row in results["per_map"]:
                self.assertGreaterEqual(row["optimality_ratio"], 1.0 - 1e-12)
                if row["found"] and row["optimal_path_cost"] > 0.0:
                    self.assertAlmostEqual(
                        row["optimality_ratio"],
                        row["found_ground_truth_cost"] / row["optimal_path_cost"],
                    )
                if row["method"].startswith("SeeA*"):
                    self.assertGreaterEqual(
                        row["search_time_seconds"],
                        row["sampling_overhead_seconds"],
                    )


if __name__ == "__main__":
    unittest.main()
