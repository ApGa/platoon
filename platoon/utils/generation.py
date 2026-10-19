"""The trajectory misc key that persists each model call's sampling diagnostics.

``platoon.gateway.trace`` stores, per completion id, the call's stop and parse
diagnostics (``ChatResponse.generation``) plus its finish reason under
``MODEL_GENERATIONS_KEY`` in the trajectory's misc, so they survive message
conversion and reach error filtering, metrics and the trajectory viewer.
"""

MODEL_GENERATIONS_KEY = "model_generations"
