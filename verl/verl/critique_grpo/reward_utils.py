"""Reward routing shared by Critique-GRPO rollout selection and training."""

from verl.utils.reward_score import _default_compute_score


def compute_score(data_source: str, solution_str: str, ground_truth: str) -> dict:
    """Return a score dictionary using the evaluator selected by data_source."""
    result = _default_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
    )
    if isinstance(result, dict):
        return result
    score = float(result)
    return {"score": score, "acc": score}
