"""Regression tests for Critique-GRPO's Math-Verify execution context."""

import ast
import importlib.util
from pathlib import Path
import runpy
import unittest


REPO_VERL_ROOT = Path(__file__).parents[2]


def _thread_pool_calls(path: Path):
    tree = ast.parse(path.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ThreadPoolExecutor"
    ]


class CritiqueMathRewardExecutionTest(unittest.TestCase):
    def test_reward_call_sites_do_not_use_thread_pool(self):
        paths = (
            REPO_VERL_ROOT / "verl" / "critique_grpo" / "critique_main_ppo.py",
            REPO_VERL_ROOT / "verl" / "critique_grpo" / "critique_vllm_rollout_spmd.py",
        )

        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(_thread_pool_calls(path), [])

    @unittest.skipUnless(importlib.util.find_spec("math_verify"), "math-verify is not installed")
    def test_math_reward_accepts_a_correct_boxed_answer(self):
        module = runpy.run_path(
            REPO_VERL_ROOT / "verl" / "utils" / "reward_score" / "think_test_math.py"
        )

        result = module["compute_score"](r"\boxed{2}", r"\boxed{2}")

        self.assertEqual(result["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
