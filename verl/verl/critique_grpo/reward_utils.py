"""Reward routing shared by Critique-GRPO rollout selection and training.

The Critique-GRPO recipe uses the same dataset identifiers as DenoiseRL:
``option_MMLU-Pro`` for MMLU-Pro, ``bbeh`` for BBEH, and ``think_*`` for
the math datasets. Keep this routing local to the recipe so changes to VERL's
generic reward dispatcher cannot silently change the experiment's rewards.
"""


def _select_compute_score(data_source: str):
    normalized_source = str(data_source).strip().casefold().replace("_", "-")

    if normalized_source == "bbeh":
        from verl.utils.reward_score.bbeh import compute_score

        return compute_score

    if normalized_source in {"mmlu-pro", "option-mmlu-pro"} or normalized_source.endswith("/mmlu-pro"):
        from verl.utils.reward_score.choice_base_problems import compute_score

        return compute_score

    # The remaining datasets in the Critique-GRPO recipe are math datasets
    # (think_7500, think_MATH-500, think_aime*, and think_amc*).
    from verl.utils.reward_score.think_test_math import compute_score

    return compute_score


def compute_score(data_source: str, solution_str: str, ground_truth: str) -> dict:
    """Return a score dictionary using the DenoiseRL evaluator for the dataset."""
    evaluator = _select_compute_score(data_source)
    result = evaluator(solution_str, ground_truth)
    if isinstance(result, dict):
        return result
    score = float(result)
    return {"score": score, "acc": score}
