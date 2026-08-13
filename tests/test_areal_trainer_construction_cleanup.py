from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAINER = REPO_ROOT / "platoon/train/areal/rl.py"


def test_trainer_construction_failure_synchronously_cleans_scheduler_roles():
    tree = ast.parse(TRAINER.read_text())
    trainer = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PlatoonArealRLTrainer"
    )
    init = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    guard = next(node for node in ast.walk(init) if isinstance(node, ast.Try))
    assert any(
        isinstance(handler.type, ast.Name)
        and handler.type.id == "BaseException"
        and any(
            isinstance(call, ast.Call)
            and ast.unparse(call.func) == "scheduler.delete_workers"
            and {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords} == {"role": None, "reverse_order": True}
            for call in ast.walk(handler)
        )
        for handler in guard.handlers
    )


def test_cleanup_failure_is_fail_closed_and_chained():
    tree = ast.parse(TRAINER.read_text())
    cleanup_raise = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and ast.unparse(node.exc.func) == "RuntimeError"
        and len(node.exc.args) == 1
        and isinstance(node.exc.args[0], ast.Constant)
        and node.exc.args[0].value == "Trainer construction failed and scheduler cleanup could not be confirmed"
    )
    assert isinstance(cleanup_raise.cause, ast.Name)
    assert cleanup_raise.cause.id == "cleanup_error"
