"""Math-Verify scoring used by the DenoiseRL train and math eval sets."""

from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig


def compute_score(solution_str, ground_truth):
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    acc = 0
    try:
        acc, _ = verify_func([ground_truth], [solution_str])
    except TimeoutException:
        pass
    except Exception as exc:
        # Math-Verify uses signal.alarm() for parsing timeouts and therefore
        # raises when called from a ThreadPoolExecutor. Do not turn evaluator
        # infrastructure failures into an indistinguishable incorrect answer.
        raise RuntimeError(
            "Math-Verify scoring failed. Run think_test_math.compute_score "
            "outside ThreadPoolExecutor worker threads."
        ) from exc

    reward = 1.0 if acc else 0.0
    return {
        "score": reward,
        "acc": acc,
        "answer": solution_str,
        "pred": "",
        "format_verify": 0.0,
    }
