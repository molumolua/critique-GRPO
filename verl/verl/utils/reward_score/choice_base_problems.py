"""Multiple-choice evaluator aligned with DenoiseRL."""

import re

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import StringExtractionConfig

    _HAS_MATH_VERIFY = True
except ImportError:
    TimeoutException = Exception
    _HAS_MATH_VERIFY = False


def _normalize_choice(value: str):
    if value is None:
        return None
    value = value.strip()

    match = re.search(r"\\boxed\{([^}]*)\}", value)
    if match:
        inner = match.group(1).strip()
        if re.fullmatch(r"[A-Za-z]", inner):
            return inner.casefold()
        value = inner

    match = re.fullmatch(r"[\s\(\[\{\<]*([A-Za-z])[\s\)\]\}\>]*", value)
    if match:
        return match.group(1).casefold()

    letters = re.findall(r"\b([A-Za-z])\b", value)
    if len(letters) == 1:
        return letters[0].casefold()
    return None


def compute_score(solution_str: str, ground_truth: str):
    acc = 0
    gt_norm = _normalize_choice(ground_truth)
    pred_norm = _normalize_choice(solution_str)

    if gt_norm is not None and pred_norm is not None:
        acc = int(gt_norm == pred_norm)
    elif _HAS_MATH_VERIFY:
        verify_func = math_metric(
            gold_extraction_target=(StringExtractionConfig(),),
            pred_extraction_target=(StringExtractionConfig(),),
        )
        try:
            acc, _ = verify_func([ground_truth], [solution_str])
        except (Exception, TimeoutException):
            pass

    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "answer": solution_str,
        "pred": str(pred_norm),
        "format_verify": 0.0,
    }
