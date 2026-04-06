from __future__ import annotations

import re
from typing import Final


FINAL_ANSWER_DELIMITER: Final[str] = "####"
NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def normalize_answer(text: str) -> str:
    normalized = text.strip()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace(",", "")
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def extract_final_answer_from_reference(answer_text: str) -> tuple[str, str]:
    """Split GSM8K reference answer into CoT and final answer."""
    parts = answer_text.split(FINAL_ANSWER_DELIMITER, maxsplit=1)
    if len(parts) == 2:
        cot, final_answer = parts
        return cot.strip(), normalize_answer(final_answer)
    return answer_text.strip(), normalize_answer(answer_text)


def extract_final_answer_from_model_output(text: str) -> str:
    """Best-effort final answer extraction from model output."""
    if FINAL_ANSWER_DELIMITER in text:
        return normalize_answer(text.split(FINAL_ANSWER_DELIMITER)[-1])

    labeled_patterns = [
        r"Final Answer\s*:\s*(.+)",
        r"Answer\s*:\s*(.+)",
        r"The answer is\s*(.+)",
    ]
    for pattern in labeled_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return normalize_answer(match.group(1))

    matches = NUMBER_PATTERN.findall(text)
    if matches:
        return normalize_answer(matches[-1])

    return normalize_answer(text)


def split_reasoning_and_answer(text: str) -> tuple[str, str]:
    final_answer = extract_final_answer_from_model_output(text)
    if FINAL_ANSWER_DELIMITER in text:
        reasoning = text.split(FINAL_ANSWER_DELIMITER, maxsplit=1)[0].strip()
        return reasoning, final_answer

    labeled_match = re.search(
        r"(.*?)(?:Final Answer|Answer)\s*:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if labeled_match:
        return labeled_match.group(1).strip(), final_answer

    return text.strip(), final_answer
