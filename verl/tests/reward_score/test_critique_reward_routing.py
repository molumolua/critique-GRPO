"""Tests for the dataset-specific Critique-GRPO reward routing."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


class CritiqueRewardRoutingTest(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.module_patchers = []

        for module_name, label in (
            ("verl.utils.reward_score.bbeh", "bbeh"),
            ("verl.utils.reward_score.choice_base_problems", "choice"),
            ("verl.utils.reward_score.think_test_math", "math"),
        ):
            module = ModuleType(module_name)

            def compute_score(solution_str, ground_truth, evaluator=label):
                self.calls.append((evaluator, solution_str, ground_truth))
                return {"score": 1.0, "acc": 1}

            module.compute_score = compute_score
            patcher = patch.dict(sys.modules, {module_name: module})
            patcher.start()
            self.module_patchers.append(patcher)

        module_path = (
            Path(__file__).parents[2]
            / "verl"
            / "critique_grpo"
            / "reward_utils.py"
        )
        spec = importlib.util.spec_from_file_location("critique_reward_utils_under_test", module_path)
        self.reward_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.reward_utils)

    def tearDown(self):
        for patcher in reversed(self.module_patchers):
            patcher.stop()

    def assert_route(self, data_source, expected_evaluator, ground_truth):
        self.reward_utils.compute_score(data_source, "response", ground_truth)
        self.assertEqual(self.calls, [(expected_evaluator, "response", ground_truth)])

    def test_mmlu_pro_uses_choice_evaluator(self):
        for data_source in ("option_MMLU-Pro", "mmlu-pro", "TIGER-Lab/MMLU-Pro"):
            with self.subTest(data_source=data_source):
                self.calls.clear()
                self.assert_route(data_source, "choice", "A")

    def test_bbeh_uses_bbeh_evaluator(self):
        self.assert_route("bbeh", "bbeh", "answer")

    def test_other_recipe_datasets_use_math_evaluator(self):
        for data_source in ("think_7500", "think_MATH-500", "think_aime25", "think_amc23"):
            with self.subTest(data_source=data_source):
                self.calls.clear()
                self.assert_route(data_source, "math", r"\boxed{2}")


if __name__ == "__main__":
    unittest.main()
