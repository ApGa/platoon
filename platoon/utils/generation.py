"""Small, persisted sampling diagnostics shared by providers and agent harnesses."""

from typing import Any

MODEL_GENERATIONS_KEY = "model_generations"


def generation_for_completion(completion_id: str | None) -> dict[str, Any]:
    from platoon.episode.context import current_trajectory

    trajectory = current_trajectory.get(None)
    if trajectory is None:
        return {}
    return trajectory.misc.get(MODEL_GENERATIONS_KEY, {}).get(completion_id, {})


def record_generation(response: Any) -> dict[str, Any]:
    """Keep stop information even when SDK message conversion drops it.

    Tinker supplies raw stop/parse diagnostics; other OpenAI-compatible
    providers still expose the finish reason and completion token count.
    """
    from platoon.episode.context import current_trajectory

    raw = getattr(response, "raw_response", response)
    choices = getattr(raw, "choices", None)
    if not choices:
        return {}
    choice = choices[0]
    metadata = dict(getattr(choice, "platoon_generation", None) or {})
    metadata["finish_reason"] = getattr(choice, "finish_reason", None)
    if "generated_tokens" not in metadata:
        metadata["generated_tokens"] = getattr(getattr(raw, "usage", None), "completion_tokens", None)
    trajectory = current_trajectory.get(None)
    completion_id = getattr(raw, "id", None)
    if trajectory is not None and completion_id is not None:
        generations = trajectory.misc.setdefault(MODEL_GENERATIONS_KEY, {})
        generations.setdefault(completion_id, {}).update(metadata)
        return generations[completion_id]
    return metadata
