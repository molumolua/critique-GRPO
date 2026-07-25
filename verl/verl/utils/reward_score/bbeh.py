"""BigBench Extra Hard evaluator aligned with DenoiseRL."""

import re


def _last_boxed_only_string(value):
    index = value.rfind("\\boxed")
    if "\\boxed " in value:
        return "\\boxed " + value.split("\\boxed ")[-1].split("$")[0]
    if index < 0:
        index = value.rfind("\\fbox")
        if index < 0:
            return None

    opened = 0
    for cursor in range(index, len(value)):
        if value[cursor] == "{":
            opened += 1
        elif value[cursor] == "}":
            opened -= 1
            if opened == 0:
                return value[index : cursor + 1]
    return None


def _strip_latex(value: str) -> str:
    if value.startswith("$") and value.endswith("$"):
        value = value[1:-1]
    for wrapper in ("boxed{", "text{", "texttt{"):
        if wrapper in value and value.endswith("}"):
            value = value[:-1].split(wrapper)[-1]
    return value


def _extract_answer(sample: str):
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", sample, re.DOTALL)
    matched = bool(matches)
    output = matches[-1].strip() if matches else _last_boxed_only_string(sample)
    if output is None:
        output = sample
    else:
        matched = True

    for prefix in (
        "The answer is:",
        "The final answer is ",
        "The final answer is: ",
        "The answer is ",
    ):
        output = output.replace(prefix, "").strip()
    if output.endswith("."):
        output = output[:-1].strip()
    return _strip_latex(output).lower(), matched


def _fuzzy_match(prediction: str, reference: str) -> bool:
    prediction = str(prediction).lower()
    reference = str(reference).lower()
    if prediction == reference:
        return True
    if len(prediction) == 3 and prediction[0] == "(" and prediction[-1] == ")":
        return prediction[1] == reference
    if len(reference) == 3 and reference[0] == "(" and reference[-1] == ")":
        return reference[1] == prediction
    try:
        if float(prediction) == float(reference):
            return True
    except ValueError:
        pass
    if prediction.replace("'", "") == reference.replace("'", ""):
        return True
    if f"[{reference}]" == prediction or f"[{prediction}]" == reference:
        return True
    return prediction.endswith("?") and prediction[:-1] == reference


def compute_score(predict_str: str, ground_truth: str):
    prediction, matched = _extract_answer(predict_str.strip())
    prediction = prediction.replace(", ", ",").replace("**", "").split("\n")[0]
    prediction = prediction[:-1] if prediction.endswith(".") else prediction
    if not matched:
        boxed = _last_boxed_only_string(prediction)
        if boxed:
            prediction = boxed

    reference = ground_truth.strip().lower().replace(", ", ",")
    acc = int(_fuzzy_match(prediction, reference))
    return {
        "score": 1.0 if acc else -1.0,
        "acc": acc,
        "answer": predict_str,
        "pred": prediction,
        "format_verify": 0.0,
    }
