"""Execution spans recorded for critical-path attribution.

Each trajectory records when it ran (``EXECUTION_SPAN_KEY``) and when each of
its model calls ran (``MODEL_CALL_SPANS_KEY``) in its ``misc``.  Timestamps
establish ordering only, so consumers such as a critical-path token
efficiency reward can tell parallel children from sequential ones.
"""

EXECUTION_SPAN_KEY = "execution_span"
MODEL_CALL_SPANS_KEY = "model_call_spans"
