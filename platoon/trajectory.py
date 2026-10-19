from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

from platoon.task import Task


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # Walk dataclass fields WITHOUT dataclasses.asdict(): asdict() deep-copies
    # every leaf value, and trajectory steps embed live SDK objects (e.g.
    # OpenHands events holding a threading.Lock / asyncio.Future) that cannot be
    # copied/pickled. Recursing via _to_jsonable instead routes those through
    # model_dump()/str() below rather than copy.deepcopy.
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_jsonable(getattr(obj, f.name)) for f in fields(obj)}
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        try:
            return model_dump(mode="json")
        except TypeError:
            try:
                return _to_jsonable(model_dump())
            except Exception:
                return str(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(x) for x in obj]
    return str(obj)


@dataclass
class TrajectoryStep:
    misc: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentInfo:
    id: str
    fork_step: int


@dataclass
class Trajectory:
    """One agent's record in a rollout: its task, its steps (one per recorded model call), reward and outcome."""

    id: str
    # The agent's task.
    task: Task | None = None
    # The parent trajectory and the step it spawned this agent at (None for the root).
    parent_info: ParentInfo | None = None
    # One step per recorded model call; each step keeps its data in ``misc``.
    steps: list[TrajectoryStep] = field(default_factory=list)
    reward: float = 0.0
    # The agent's final message.
    finish_message: str | None = None
    # Why the agent failed, if it did.
    error_message: str | None = None
    # Agent-level annotations, such as the outcome and the harness's ``ctx.annotate`` data.
    misc: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryCollection:
    """Every trajectory of one rollout: the root agent's and its subagents', by trajectory id."""

    # The rollout's id; a new UUID by default.
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # The rollout's trajectories by id, the root's first.
    trajectories: dict[str, Trajectory] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for portability, normalizing nested SDK objects such as OpenHands events into JSON-safe
        dictionaries."""
        return {
            "id": self.id,
            "trajectories": {traj_id: _to_jsonable(traj) for traj_id, traj in self.trajectories.items()},
        }
