from __future__ import annotations

import re


def classify_failure(text: str, timeout: bool = False) -> str:
    if timeout:
        return "TIMEOUT"
    normalized = text.lower()
    patterns = [
        ("CUDA_OOM", r"out of memory|cuda.*memory|cublas_status_alloc_failed"),
        ("NAN_LOSS", r"nan|not a number|non[- ]finite"),
        ("SHAPE_ERROR", r"shape mismatch|dimension|size mismatch|expected .* got"),
        ("IMPORT_ERROR", r"module not found|importerror|no module named"),
        ("DATA_ERROR", r"filenotfounderror|keyerror|dataset|column.*not found"),
        ("SYNTAX_ERROR", r"syntaxerror|indentationerror"),
        ("METRIC_MISSING", r"metrics.*not found|metric.*missing|required output"),
        ("TEST_FAILED", r"failed.*test|assertionerror|pytest"),
    ]
    for code, pattern in patterns:
        if re.search(pattern, normalized):
            return code
    return "PROCESS_FAILED"
