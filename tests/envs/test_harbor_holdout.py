"""Held-out splits by whole groups (``service_kwargs.holdout``): the rule over a dataset's selected tasks, and the
``train`` and ``eval`` splits a Harbor service lists with it, from a lock or a dataset directory."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from platoon.envs.harbor import HarborEnvironmentService, inputs
from platoon.envs.harbor.fetch import answering_lock, current_lock, ensure_lock, fetch
from platoon.envs.harbor.holdout import HoldoutError, HoldoutRow, HoldoutRule, hold_out
from platoon.envs.harbor.inputs import EMPTY_FILE, copied_paths, input_digest, input_files, normalized_text
from platoon.envs.harbor.lock import read_lock, write_lock
from platoon.envs.service import EnvSetupError
from platoon.train.auto import AutoDataset
from platoon.train.components import EnvironmentConfig


def _rule(**values) -> HoldoutRule:
    return HoldoutRule.coerce({"fraction": 0.25, "group_by": "group", **values})


def _row(task_id: str, fields: dict, refused: str | None = None, input_files: dict | None = None) -> HoldoutRow:
    """A task with ``input_files`` (none by default)."""

    return HoldoutRow(task_id, fields, refused, {} if input_files is None else input_files)


def _rows(groups: dict[str, int], *, stratum: str | None = None, prefix: str = "") -> list[HoldoutRow]:
    """``count`` tasks per group, named ``<prefix><group>-<n>``, each with an input file of its own."""

    fields = {} if stratum is None else {"source": stratum}
    return [
        _row(
            f"{prefix}{group}-{n}",
            {"group": group, **fields},
            input_files={"environment/input/data.csv": _file(f"{prefix}{group}-{n}")},
        )
        for group, count in groups.items()
        for n in range(count)
    ]


def _file(content: str) -> str:
    """The digest of a file holding ``content``."""

    return "sha256:" + hashlib.sha256(content.encode()).hexdigest()


def _text(content: str) -> str:
    """The input digest of a text file holding ``content``, one line or several, without blank lines or blanks at a
    line's end: the SHA-256 of the text with its last line ended."""

    return _file(content if content.endswith("\n") else content + "\n")


def _groups_of(task_ids, rows: list[HoldoutRow]) -> set[str]:
    by_id = {row.task_id: row for row in rows}
    return {by_id[task_id].fields["group"] for task_id in task_ids}


def _rank(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}\0{name}".encode()).hexdigest()


# --- the rule ----------------------------------------------------------------------------------------------------


def test_the_same_rows_and_rule_give_the_same_split_in_any_order():
    rows = _rows({f"g{n}": 1 + n % 4 for n in range(40)})
    rule = _rule()
    first = hold_out(rows, rule)
    shuffled = rows[:]
    random.Random(7).shuffle(shuffled)

    second = hold_out(shuffled, rule)

    assert (first.train, first.eval) == (second.train, second.eval)
    assert first.eval_digest == second.eval_digest


def test_groups_are_never_split_and_each_stratum_holds_out_at_most_its_share():
    rows = _rows({f"a{n}": 1 + n % 5 for n in range(30)}, stratum="a") + _rows(
        {f"b{n}": 1 + n % 3 for n in range(30)}, stratum="b", prefix="b-"
    )
    for seed in range(10):
        split = hold_out(rows, _rule(stratify_by="source", seed=seed))
        held_groups = _groups_of(split.eval, rows)
        for group in held_groups:  # every task of a held-out group is held out, none trains
            members = {row.task_id for row in rows if row.fields["group"] == group}
            assert members <= set(split.eval) and not members & set(split.train)
        for stratum in split.strata:
            selected = sum(1 for row in rows if row.fields["source"] == stratum.stratum)
            assert stratum.selected == selected and stratum.held_out <= -(-selected // 4)  # ceil(0.25 * selected)
        assert set(split.train) | set(split.eval) == {row.task_id for row in rows}
        assert not set(split.train) & set(split.eval)


def test_a_group_with_tasks_in_two_strata_is_held_out_whole_or_not_at_all():
    """A group may have tasks in two strata, such as a data file that two sources ask about. It must fit the share of
    each, and is held out in both or in neither, whatever the seed: walking each stratum on its own would hold it out
    where it fits and train on it where it does not."""

    rows = [_row("easy-shared-0", {"group": "shared", "source": "easy"})]
    rows += [_row(f"easy-{n}", {"group": f"e{n}", "source": "easy"}) for n in range(9)]
    rows += [_row(f"hard-shared-{n}", {"group": "shared", "source": "hard"}) for n in range(3)]
    rows += [_row(f"hard-{n}", {"group": f"h{n}", "source": "hard"}) for n in range(9)]
    shared = {row.task_id for row in rows if row.fields["group"] == "shared"}
    held_whole = 0
    for seed in range(200):
        split = hold_out(rows, _rule(fraction=0.2, stratify_by="source", seed=seed))
        assert shared <= set(split.eval) or shared <= set(split.train), seed
        held_whole += shared <= set(split.eval)
        shares = {"easy": 2, "hard": 3}  # ceil(0.2 * 10) and ceil(0.2 * 12)
        assert all(stratum.held_out <= shares[str(stratum.stratum)] for stratum in split.strata), seed
    assert 0 < held_whole < 200  # it fits both shares for some seeds only


def test_a_group_that_does_not_fit_is_passed_over():
    """The stratum's share is 2 tasks: the 5-task group never fits, so two single-task groups fill the share, in
    whatever order the seed ranks the groups."""

    rows = _rows({"big": 5, "s1": 1, "s2": 1, "s3": 1})
    for seed in range(20):
        split = hold_out(rows, _rule(fraction=0.2, seed=seed))  # ceil(0.2 * 8) = 2
        assert len(split.eval) == 2 and "big" not in _groups_of(split.eval, rows)
        ranked = sorted(["big", "s1", "s2", "s3"], key=lambda group: _rank(seed, group))
        assert _groups_of(split.eval, rows) == set([group for group in ranked if group != "big"][:2])


def test_the_share_is_computed_on_the_decimal_the_config_wrote():
    rows = _rows({f"g{n}": 1 for n in range(100)})

    split = hold_out(rows, _rule(fraction=0.07))

    assert len(split.eval) == 7  # 0.07 * 100 is 7.000000000000001 in floating point, whose ceiling is 8


def test_refused_tasks_take_part_in_the_walk_and_leave_both_lists():
    rows = _rows({f"g{n}": 2 for n in range(20)})
    refused = [replace(row, refused="uploads_over_mb: 120 MiB") if row.task_id.endswith("-1") else row for row in rows]

    plain, with_refusals = hold_out(rows, _rule()), hold_out(refused, _rule())

    assert _groups_of(plain.eval, rows) == _groups_of(with_refusals.eval, rows)  # no group moved
    assert with_refusals.held_out == plain.held_out == plain.eval  # every held-out task, refused ones included
    assert not any(task_id.endswith("-1") for task_id in with_refusals.train + with_refusals.eval)
    (stratum,) = with_refusals.strata
    assert (stratum.selected, stratum.runnable, stratum.refused) == (40, 20, {"uploads_over_mb": 20})
    assert stratum.held_out == len(plain.eval) and stratum.eval == len(plain.eval) // 2


def test_refusals_never_move_a_group_whatever_the_seed():
    """A group larger than the share sits beside single tasks: refusing half of it would let it fit if refused tasks
    left before the walk.  They take part in it, so for every seed the same tasks are held out with or without the
    refusals, less the refused ones."""

    rows = _rows({"big": 4, **{f"s{n}": 1 for n in range(8)}})
    refused = [
        replace(row, refused="uploads_over_mb: 1 GiB") if row.task_id in ("big-0", "big-1") else row for row in rows
    ]
    for seed in range(20):
        plain, with_refusals = hold_out(rows, _rule(seed=seed)), hold_out(refused, _rule(seed=seed))
        assert set(with_refusals.eval) == set(plain.eval) - {"big-0", "big-1"}, seed
        assert with_refusals.held_out == plain.held_out, seed


def test_group_by_takes_the_first_field_a_task_has():
    rows = [
        _row("q1", {"kaggle_dataset_name": "sales", "split_group": "nb1_qa_1"}),
        _row("q2", {"kaggle_dataset_name": "sales", "split_group": "nb2_qa_1"}),
        _row("m1", {"split_group": "family-a"}),
        _row("m2", {"split_group": "family-a"}),
    ]
    rule = HoldoutRule.coerce({"fraction": 0.5, "group_by": ["kaggle_dataset_name", "split_group"]})

    split = hold_out(rows, rule)

    assert set(split.eval) in ({"q1", "q2"}, {"m1", "m2"})  # two groups of two: sales, and family-a
    assert (split.strata[0].eval_groups, split.strata[0].largest_group) in ((1, "sales"), (1, "family-a"))


def test_a_runnable_task_without_a_group_field_is_an_error_and_a_refused_one_sits_out():
    rows = _rows({"g": 3}) + [_row("bare", {"other": 1})]
    with pytest.raises(HoldoutError, match=r"1 runnable task\(s\) have none of the index fields \['group'\]"):
        hold_out(rows, _rule())

    rows[-1] = HoldoutRow("bare", {"other": 1}, "task_config: broken task.toml")  # inputs unknown: it sits out
    split = hold_out(rows, _rule())
    assert split.sat_out == 1 and "bare" not in split.train + split.eval


def test_a_runnable_task_without_the_stratum_field_is_an_error_and_a_refused_one_sits_out():
    rows = _rows({"g": 3}, stratum="a") + [_row("loose", {"group": "g"})]
    with pytest.raises(HoldoutError, match="have no index field 'source' to stratify by, such as \\['loose'\\]"):
        hold_out(rows, _rule(stratify_by="source"))

    rows[-1] = _row("loose", {"group": "g"}, "gpus: the task asks for 1 GPU(s)")
    assert hold_out(rows, _rule(stratify_by="source")).sat_out == 1


def test_the_seed_changes_the_split():
    rows = _rows({f"g{n}": 1 for n in range(50)})

    assert hold_out(rows, _rule(seed=0)).eval != hold_out(rows, _rule(seed=1)).eval


def test_the_lists_come_in_hash_order_and_the_digest_covers_the_sorted_eval_ids():
    rows = _rows({f"g{n}": 2 for n in range(30)})
    rule = _rule(seed=3)

    split = hold_out(rows, rule)

    assert list(split.eval) == sorted(split.eval, key=lambda task_id: _rank(3, task_id))
    assert list(split.train) == sorted(split.train, key=lambda task_id: _rank(3, task_id))
    assert split.eval_digest == hashlib.sha256("\n".join(sorted(split.eval)).encode()).hexdigest()


def test_the_report_gives_each_strata_makeup():
    rows = _rows({"x1": 3, "x2": 1, "x3": 1, "x4": 1}, stratum="smol") + _rows({"y1": 4}, stratum="mco", prefix="m-")
    rows[0] = replace(rows[0], refused="uploads_over_mb: 200 MiB")
    split = hold_out(rows, HoldoutRule.coerce({"fraction": 0.5, "group_by": "group", "stratify_by": "source"}))

    by_name = {stratum.stratum: stratum for stratum in split.strata}
    assert [stratum.stratum for stratum in split.strata] == ["mco", "smol"]
    assert (by_name["smol"].selected, by_name["smol"].runnable, by_name["smol"].refused) == (
        6,
        5,
        {"uploads_over_mb": 1},
    )
    # mco's share is ceil(0.5 * 4) = 2 tasks, and its one group has 4: it does not fit, so mco holds out nothing.
    assert (by_name["mco"].held_out, by_name["mco"].eval, by_name["mco"].train) == (0, 0, 4)
    lines = split.report()
    assert lines[0] == "hold-out: 50% of each source, grouped by group, joined by shared input files, seed 0"
    assert lines[2].startswith("  source smol: 6 selected, 5 runnable; held out ")
    assert "refused: uploads_over_mb 1" in lines[2]
    assert lines[3] == "  groups: 5 by group; no two of them share an input file; largest y1: 4 task(s)"
    assert lines[-1].endswith(f"eval digest {split.eval_digest[:12]}")
    assert split.to_json()["strata"][1]["refused"] == {"uploads_over_mb": 1}


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"group_by": "g"}, "holdout.fraction must be a number between 0 and 1, got None"),
        ({"fraction": 0, "group_by": "g"}, "holdout.fraction must be a number between 0 and 1"),
        ({"fraction": 1, "group_by": "g"}, "holdout.fraction must be a number between 0 and 1"),
        ({"fraction": "0.1", "group_by": "g"}, "holdout.fraction must be a number between 0 and 1"),
        ({"fraction": True, "group_by": "g"}, "holdout.fraction must be a number between 0 and 1"),
        ({"fraction": 0.1}, "holdout.group_by must name an index field"),
        ({"fraction": 0.1, "group_by": []}, "holdout.group_by must name an index field"),
        ({"fraction": 0.1, "group_by": ["g", ""]}, "holdout.group_by must name an index field"),
        ({"fraction": 0.1, "group_by": "g", "stratify_by": ""}, "holdout.stratify_by must name an index field"),
        ({"fraction": 0.1, "group_by": "g", "seed": True}, "holdout.seed must be an integer"),
        ({"fraction": 0.1, "group_by": "g", "seed": "0"}, "holdout.seed must be an integer"),
        ({"fraction": 0.1, "group_by": "g", "groups": 3}, r"holdout has unknown keys \['groups'\]"),
        (0.1, "holdout must be a mapping"),
    ],
)
def test_rules_are_checked(value, message):
    with pytest.raises(HoldoutError, match=message):
        HoldoutRule.coerce(value)


# --- shared input files -------------------------------------------------------------------------------------------


def _singles(count: int, *, prefix: str = "s") -> list[HoldoutRow]:
    """``count`` one-task groups, each with an input file of its own."""

    own = "environment/input/own.csv"
    return [
        _row(f"{prefix}{n}", {"group": f"{prefix}{n}"}, input_files={own: _file(f"{prefix}{n}")}) for n in range(count)
    ]


def _together(split, task_ids: set[str]) -> bool:
    return task_ids <= set(split.eval) or task_ids <= set(split.train)


def test_two_groups_whose_tasks_share_one_input_file_are_one_group():
    """Groups a and b ask about different datasets by their fields, but a task of each holds the same table (under
    another name): for every seed both land on the same side, as one group named after a, the larger.  Without the
    shared file, some seeds split them."""

    rows = [_row(f"a-{n}", {"group": "a"}, input_files={"environment/a.csv": _file(f"a{n}")}) for n in range(3)]
    rows += [_row(f"b-{n}", {"group": "b"}, input_files={"environment/b.csv": _file(f"b{n}")}) for n in range(2)]
    rows += _singles(15)
    shared = [
        replace(rows[0], input_files={"environment/a.csv": _file("a0"), "environment/input/iris.csv": _file("iris")}),
        *rows[1:3],
        replace(rows[3], input_files={"environment/b.csv": _file("b0"), "environment/data/Iris.csv": _file("iris")}),
        *rows[4:],
    ]
    together = {"a-0", "a-1", "a-2", "b-0", "b-1"}

    held = sum(_together(hold_out(shared, _rule(fraction=0.5, seed=seed)), together) for seed in range(60))
    in_eval = sum(together <= set(hold_out(shared, _rule(fraction=0.5, seed=seed)).eval) for seed in range(60))
    apart = sum(not _together(hold_out(rows, _rule(fraction=0.5, seed=seed)), together) for seed in range(60))

    assert held == 60 and 0 < in_eval < 60  # a share of 10 of 20 tasks: the joined group fits for some seeds
    assert apart > 0
    groups = hold_out(shared, _rule()).groups
    assert groups is not None
    assert (groups.by_fields, groups.joined_groups, groups.joined_into, groups.groups) == (17, 2, 1, 16)
    assert (groups.largest, groups.largest_tasks, groups.largest_members) == ("a", 5, 2)
    assert (groups.largest_shared_files, groups.largest_example, groups.largest_example_tasks) == (
        1,
        "environment/data/Iris.csv",  # the first of the file's paths in sorted order
        2,
    )
    assert groups.line(_rule()) == (
        "  groups: 17 by group; 1 shared input file(s) joined 2 of them into 1, leaving 16; largest a: 5 task(s), 2 "
        "groups joined by 1 file(s) such as environment/data/Iris.csv (in 2 task(s))"
    )


def test_groups_join_through_a_chain_of_shared_files():
    """a shares one file with b, and b another with c: a and c share nothing, yet the three are one group."""

    rows = [
        _row("a-0", {"group": "a"}, input_files={"environment/x.csv": _file("x")}),
        _row("b-0", {"group": "b"}, input_files={"environment/x.csv": _file("x"), "environment/y.csv": _file("y")}),
        _row("c-0", {"group": "c"}, input_files={"environment/y.csv": _file("y")}),
        _row("c-1", {"group": "c"}, input_files={"environment/z.csv": _file("z")}),
        *_singles(12),
    ]
    chain = {"a-0", "b-0", "c-0", "c-1"}

    for seed in range(60):
        assert _together(hold_out(rows, _rule(fraction=0.5, seed=seed)), chain), seed
    assert any(chain <= set(hold_out(rows, _rule(fraction=0.5, seed=seed)).eval) for seed in range(60))
    groups = hold_out(rows, _rule()).groups
    assert groups is not None
    assert (groups.joined_groups, groups.joined_into, groups.largest, groups.largest_members) == (3, 1, "c", 3)
    assert groups.largest_shared_files == 2 and groups.shared_files == 2


def test_input_files_join_by_their_bytes_not_by_their_names():
    """Two tasks whose files have one name and different bytes share nothing; two whose files have different names
    and the same bytes share one."""

    rows = [
        _row("a-0", {"group": "a"}, input_files={"environment/input/data.csv": _file("one")}),
        _row("b-0", {"group": "b"}, input_files={"environment/input/data.csv": _file("two")}),
        _row("c-0", {"group": "c"}, input_files={"environment/input/train.csv": _file("three")}),
        _row("d-0", {"group": "d"}, input_files={"environment/raw/copy of train.csv": _file("three")}),
        *_singles(4),
    ]

    groups = hold_out(rows, _rule()).groups
    assert groups is not None and (groups.joined_groups, groups.joined_into, groups.shared_files) == (2, 1, 1)
    for seed in range(40):
        assert _together(hold_out(rows, _rule(seed=seed)), {"c-0", "d-0"}), seed
    assert any(not _together(hold_out(rows, _rule(seed=seed)), {"a-0", "b-0"}) for seed in range(40))


def test_a_task_without_input_files_and_an_empty_file_join_nothing():
    """An empty file holds no data, so two tasks that both hold one share nothing; a task without input files keeps
    its own group."""

    rows = [
        _row("a-0", {"group": "a"}, input_files={"environment/input/.keep": EMPTY_FILE, "environment/a": _file("a")}),
        _row("b-0", {"group": "b"}, input_files={"environment/input/.keep": EMPTY_FILE}),
        _row("c-0", {"group": "c"}, input_files={}),
        *_singles(5),
    ]

    split = hold_out(rows, _rule())

    assert split.groups is not None and (split.groups.joined_groups, split.groups.distinct_files) == (0, 6)
    assert split.groups.line(_rule()).startswith("  groups: 8 by group; no two of them share an input file")
    assert any(not _together(hold_out(rows, _rule(seed=seed)), {"a-0", "b-0"}) for seed in range(40))


def test_a_task_whose_input_files_are_not_known_is_an_error():
    rows = _rows({"g": 3}) + [HoldoutRow("old", {"group": "h"})]  # input_files=None: not known, not "none"

    with pytest.raises(HoldoutError, match=r"1 task\(s\) have no record of their input files, such as \['old'\]"):
        hold_out(rows, _rule())


def test_a_refused_tasks_input_files_join_groups_and_its_refusal_moves_nothing():
    """A refused task that holds a file of a and a file of b joins them: should it run again one day, a and b must
    already be on one side.  Refusing it or not gives the same held-out tasks."""

    rows = [
        _row("a-0", {"group": "a"}, input_files={"environment/a.csv": _file("a")}),
        _row("b-0", {"group": "b"}, input_files={"environment/b.csv": _file("b")}),
        _row("r-0", {"group": "r"}, input_files={"environment/a.csv": _file("a"), "environment/b.csv": _file("b")}),
        *_singles(9),
    ]
    refused = [replace(row, refused="uploads_over_mb: 900 MiB") if row.task_id == "r-0" else row for row in rows]

    for seed in range(40):
        plain, with_refusal = hold_out(rows, _rule(seed=seed)), hold_out(refused, _rule(seed=seed))
        assert _together(with_refusal, {"a-0", "b-0"}), seed
        assert with_refusal.held_out == plain.held_out, seed


def test_a_joined_group_with_tasks_in_two_strata_is_held_out_whole_or_not_at_all():
    rows = [
        _row("easy-a", {"group": "a", "source": "easy"}, input_files={"environment/t.csv": _file("t")}),
        _row("hard-b", {"group": "b", "source": "hard"}, input_files={"environment/t.csv": _file("t")}),
        *[replace(row, fields={**row.fields, "source": "easy"}) for row in _singles(9, prefix="e")],
        *[replace(row, fields={**row.fields, "source": "hard"}) for row in _singles(9, prefix="h")],
    ]

    held = 0
    for seed in range(100):
        split = hold_out(rows, _rule(fraction=0.2, stratify_by="source", seed=seed))
        assert _together(split, {"easy-a", "hard-b"}), seed
        held += {"easy-a", "hard-b"} <= set(split.eval)
    assert 0 < held < 100


def _shuffled(rows: list[HoldoutRow], seed: int) -> list[HoldoutRow]:
    """``rows`` in another order, each with its input files in another order."""

    order = rows[:]
    random.Random(seed).shuffle(order)
    return [replace(row, input_files=dict(reversed(list((row.input_files or {}).items())))) for row in order]


_HASH_SEED_SCRIPT = """
import json, sys
from platoon.envs.harbor.holdout import HoldoutRow, HoldoutRule, hold_out
data = json.load(sys.stdin)
rows = [HoldoutRow(row["task_id"], row["fields"], row["refused"], row["input_files"]) for row in data["rows"]]
split = hold_out(rows, HoldoutRule.coerce(data["rule"]))
print(json.dumps({"train": split.train, "eval": split.eval, "held_out": split.held_out, "report": split.report()}))
"""


def test_the_split_is_the_same_whatever_the_order_of_rows_and_files_and_the_hash_seed():
    """Joining, naming and walking the groups depend on nothing that the order of the rows, of their input files or
    Python's string hashing (``PYTHONHASHSEED``) changes."""

    pick = random.Random(11)
    rows = [
        _row(
            f"t{n:02d}",
            {"group": f"g{n % 17}", "source": f"s{n % 3}"},
            "uploads_over_mb: 200 MiB" if n % 13 == 0 else None,
            {f"environment/f{k}.csv": _file(f"c{pick.randrange(400)}") for k in range(pick.randrange(3))},
        )
        for n in range(80)
    ]
    rule = {"fraction": 0.2, "group_by": "group", "stratify_by": "source", "seed": 3}
    reference = hold_out(rows, HoldoutRule.coerce(rule))
    assert reference.groups is not None and reference.groups.joined_groups > 0  # the files join some groups
    for seed in range(5):
        again = hold_out(_shuffled(rows, seed), HoldoutRule.coerce(rule))
        assert (again.train, again.eval, again.held_out, again.report()) == (
            reference.train,
            reference.eval,
            reference.held_out,
            reference.report(),
        )

    payload = json.dumps({"rule": rule, "rows": [row.__dict__ for row in rows]})
    expected = {
        "train": list(reference.train),
        "eval": list(reference.eval),
        "held_out": list(reference.held_out),
        "report": reference.report(),
    }
    for hash_seed in ("0", "1", "4242"):
        result = subprocess.run(
            [sys.executable, "-c", _HASH_SEED_SCRIPT],
            input=payload,
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        assert json.loads(result.stdout) == expected, hash_seed


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ({"fraction": 0.1, "group_by": "g", "ignore_inputs": [""]}, "holdout.ignore_inputs must list"),
        ({"fraction": 0.1, "group_by": "g", "ignore_inputs": 3}, "holdout.ignore_inputs must list"),
        ({"fraction": 0.1, "group_by": "g", "ignore_inputs": [None]}, "holdout.ignore_inputs must list"),
    ],
)
def test_ignore_inputs_is_checked(value, message):
    with pytest.raises(HoldoutError, match=message):
        HoldoutRule.coerce(value)


def test_files_every_task_holds_by_design_join_nothing_once_named_in_ignore_inputs():
    """Every task of a game holds its code and rules, which would make the whole dataset one group; named in
    ``ignore_inputs`` they join nothing, and the rule records them.  A pattern that matches no file is an error."""

    rows = [
        _row(
            f"t{n}",
            {"group": f"goal-{n}"},
            input_files={
                "environment/game.py": _file("game"),
                "environment/rules.json": _file("rules"),
                "environment/task.json": _file(f"goal {n}"),
            },
        )
        for n in range(12)
    ]

    whole = hold_out(rows, _rule())
    assert whole.eval == () and whole.groups is not None and whole.groups.largest_members == 12  # one group of 12
    rule = _rule(ignore_inputs=["environment/game.py", "environment/rules.*"])
    split = hold_out(rows, rule)
    assert len(split.eval) == 3 and split.groups is not None and split.groups.joined_groups == 0
    assert split.groups.ignored_files == 2 and "(2 input file(s) ignored)" in split.groups.line(rule)
    assert rule.describe().endswith(
        "joined by shared input files other than environment/game.py, environment/rules.*, seed 0"
    )
    assert rule.to_json()["ignore_inputs"] == ["environment/game.py", "environment/rules.*"]
    assert "ignore_inputs" not in _rule().to_json()  # a rule without it records what it always recorded
    with pytest.raises(HoldoutError, match=r"holdout.ignore_inputs: \['environment/rule.json'\] match no input file"):
        hold_out(rows, _rule(ignore_inputs=["environment/game.py", "environment/rule.json"]))


# --- the Harbor service ------------------------------------------------------------------------------------------

DOCKERFILE = "FROM python:3.12-slim\nRUN pip install numpy\nWORKDIR /app\nCOPY instance.json /app/instance.json\n"


def _dataset(root: Path, tasks: dict[str, dict], *, refused: tuple[str, ...] = ()) -> Path:
    """Tasks with ``task-index.json`` rows; the ``refused`` ones ask for a GPU, which a fetch refuses."""

    for name in tasks:
        files = {
            "task.toml": "[environment]\ngpus = 1\n" if name in refused else "",
            "instruction.md": f"Task {name}.\n",
            "environment/Dockerfile": DOCKERFILE,
            "environment/instance.json": json.dumps({"name": name}),
            "tests/test.sh": "echo 1 > /logs/verifier/reward.txt\n",
        }
        for relative, text in files.items():
            path = root / name / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
    rows = {name: {"task_path": name, **fields} for name, fields in tasks.items()}
    (root / "task-index.json").write_text(json.dumps(rows))
    return root


def _science_like(root: Path, *, refused: tuple[str, ...] = ()) -> Path:
    tasks: dict[str, dict] = {}
    for n in range(12):
        tasks[f"smol-{n}"] = {"source_dataset": "smol", "split_group": f"kaggle-{n % 4}"}
    for n in range(8):
        tasks[f"mco-{n}"] = {"source_dataset": "mco", "split_group": f"family-{n % 4}"}
    return _dataset(root, tasks, refused=refused)


HOLDOUT = {"fraction": 0.25, "group_by": "split_group", "stratify_by": "source_dataset"}


@pytest.mark.asyncio
async def test_a_locked_service_lists_a_train_and_an_eval_split(tmp_path: Path):
    _science_like(tmp_path / "dataset", refused=("smol-5",))
    service = HarborEnvironmentService(
        dataset=str(tmp_path / "dataset"), cache=str(tmp_path / "cache"), holdout=HOLDOUT, provider="process"
    )

    assert await service.list_splits() == ["train", "eval"]
    train, evaluate = await service.list_tasks("train"), await service.list_tasks("eval")
    assert not set(train) & set(evaluate) and "smol-5" not in train + evaluate
    assert set(train) | set(evaluate) == {f"smol-{n}" for n in range(12) if n != 5} | {f"mco-{n}" for n in range(8)}
    split = service.split_report()
    assert split is not None and {stratum.stratum for stratum in split.strata} == {"smol", "mco"}
    assert {stratum.stratum: stratum.held_out for stratum in split.strata} == {"smol": 3, "mco": 2}  # one group each
    record = service.split_record()
    assert record is not None
    assert record["holdout"] == {
        "fraction": 0.25,
        "group_by": ["split_group"],
        "stratify_by": "source_dataset",
        "seed": 0,
    }
    assert record["eval"] == sorted(evaluate) and record["dataset"]["dataset"] == str(tmp_path / "dataset")
    assert set(record["held_out"]) >= set(evaluate)  # and a held-out task the lock refuses, if one fell there
    with pytest.raises(EnvSetupError, match="a hold-out lists the splits train and eval") as error:
        await service.list_tasks("val")
    assert error.value.failure_kind == "env_config"


@pytest.mark.asyncio
async def test_a_dataset_directory_splits_the_same_way(tmp_path: Path):
    root = _science_like(tmp_path / "dataset")
    locked = HarborEnvironmentService(dataset=str(root), cache=str(tmp_path / "cache"), holdout=HOLDOUT)
    directory = HarborEnvironmentService(str(root), holdout=HOLDOUT)

    assert await directory.list_tasks("eval") == await locked.list_tasks("eval")
    assert await directory.list_tasks("train") == await locked.list_tasks("train")
    record = directory.split_record()
    assert record is not None and record["dataset"] == {"dataset_dir": str(root.resolve())}


@pytest.mark.asyncio
async def test_a_hold_out_and_its_errors_are_configuration_errors(tmp_path: Path):
    root = _dataset(tmp_path / "dataset", {"a": {"source_dataset": "s"}, "b": {"source_dataset": "s"}})
    with pytest.raises(ValueError, match="Pass holdout or splits, not both"):
        HarborEnvironmentService(str(root), holdout=HOLDOUT, splits={"train": ["a"], "eval": ["b"]})
    with pytest.raises(HoldoutError, match="holdout.fraction"):
        HarborEnvironmentService(str(root), holdout={"group_by": "split_group"})
    service = HarborEnvironmentService(str(root), holdout=HOLDOUT)  # no task has a split_group
    with pytest.raises(EnvSetupError, match="holdout: 2 runnable task.s. have none of the index fields") as error:
        await service.list_tasks("train")
    assert error.value.failure_kind == "env_config"


@pytest.mark.asyncio
async def test_an_evaluation_of_a_hold_out_that_holds_out_no_runnable_task_is_a_configuration_error(tmp_path: Path):
    """Two groups of four tasks and a share of two: nothing fits, so the eval split would be empty and an evaluation
    of it would measure nothing; training on the train split is still possible."""

    root = _dataset(tmp_path / "dataset", {f"t{n}": {"split_group": f"g{n % 2}"} for n in range(8)})
    service = HarborEnvironmentService(str(root), holdout={"fraction": 0.2, "group_by": "split_group"})

    assert len(await service.list_tasks("train")) == 8
    with pytest.raises(EnvSetupError, match=r"holdout: no runnable task is held out for evaluation \(20% of") as error:
        await service.list_tasks("eval")
    assert error.value.failure_kind == "env_config" and "raise fraction, or group by a finer field" in str(error.value)


@pytest.mark.asyncio
async def test_the_report_is_logged_once_when_the_tasks_are_listed(tmp_path: Path, caplog):
    root = _science_like(tmp_path / "dataset")
    caplog.set_level("INFO", logger="platoon.envs.harbor.service")
    for split in ("train", "eval"):  # the trainer lists each split with a service of its own
        await HarborEnvironmentService(str(root), holdout={**HOLDOUT, "seed": 11}).list_tasks(split)

    reports = [record for record in caplog.records if "hold-out: 25% of each source_dataset" in record.getMessage()]
    assert len(reports) == 1 and "eval digest" in reports[0].getMessage()


def test_through_autodataset_train_and_eval_are_disjoint_and_cover_the_runnable_tasks(tmp_path: Path):
    root = _science_like(tmp_path / "dataset", refused=("mco-7",))
    entry = EnvironmentConfig(
        service="harbor",
        service_kwargs={"dataset": str(root), "cache": str(tmp_path / "cache"), "holdout": HOLDOUT},
        exclude_tasks=["smol-0", "mco-1"],
    )
    config = SimpleNamespace(environments=[entry], train=SimpleNamespace(batch_size=None, task_sampler="auto"))
    _prepare_images(root, tmp_path / "cache")  # listing refuses tasks whose images are not built

    train = list(AutoDataset.from_config(config, "train")["task_id"])
    evaluate = list(AutoDataset.from_config(config, "eval")["task_id"])

    assert not set(train) & set(evaluate)
    runnable = {f"smol-{n}" for n in range(12)} | {f"mco-{n}" for n in range(7)}
    assert set(train) | set(evaluate) == runnable - {"smol-0", "mco-1"}


def _prepare_images(dataset: Path, cache: Path, provider: str = "docker") -> None:
    """What platoon prepare does for the images of a fetched dataset, which a run checks when it lists its tasks:
    the dataset's lock fetched, and each image its tasks build recorded in the default image map (as ``local``)."""

    from platoon.envs.harbor.builders import plan_images
    from platoon.envs.harbor.fetch import fetch, fetch_config
    from platoon.envs.harbor.images import ImageEntry, default_image_map, update_image_map

    lock = fetch(fetch_config(str(dataset), cache=cache))
    entries = {image.key: ImageEntry(image="local") for image in plan_images(lock).images}
    update_image_map(default_image_map(cache, provider), entries)


# --- input files: what a task's agents' sandbox takes from it, and where a hold-out gets them ----------------------


def _task(root: Path, dockerfile: str | None, files: dict[str, str], toml: str = "") -> Path:
    contents = {"task.toml": toml, "instruction.md": "Answer.\n", "tests/test.sh": "echo 1\n", **files}
    if dockerfile is not None:
        contents["environment/Dockerfile"] = dockerfile
    for relative, text in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_input_files_are_what_the_image_takes_from_the_task_however_it_is_built(tmp_path: Path):
    """The files a Dockerfile copies from its context are the task's inputs, whether a build bakes them into the
    image (a COPY before a RUN) or a session uploads them (a deferred COPY), so moving a COPY behind a RUN changes
    nothing.  A file of the context that no step takes, the Dockerfile, the tests and the solution are not inputs."""

    files = {
        "environment/data/train.csv": "a,b\n",
        "environment/data/nested/test.csv": "c,d\n",
        "environment/question.json": "{}",
        "environment/setup.sh": "true\n",
        "environment/pull_bucket.py": "print('a helper no step copies')\n",
        "tests/expected.json": "{}",
        "solution/solve.sh": "echo\n",
    }
    deferred = (
        "FROM python:3.12-slim\nCOPY setup.sh /opt/setup.sh\nRUN sh /opt/setup.sh\nWORKDIR /app\n"
        "COPY data/ /app/data/\nCOPY question.json /app/\n"
    )
    built = (
        "FROM python:3.12-slim\nCOPY setup.sh /opt/setup.sh\nWORKDIR /app\nCOPY data/ /app/data/\n"
        "COPY question.json /app/\nRUN sh /opt/setup.sh\n"
    )
    late = _task(tmp_path / "late", deferred, files)
    early = _task(tmp_path / "early", built, files)

    expected = {
        relative: _text(files[relative])
        for relative in (
            "environment/data/nested/test.csv",
            "environment/data/train.csv",
            "environment/question.json",
            "environment/setup.sh",
        )
    }
    assert input_files(late) == input_files(early) == expected
    assert list(input_files(late)) == sorted(expected)
    assert copied_paths(late) == ("setup.sh", "data", "question.json")


def test_a_prebuilt_task_gives_its_environment_and_an_unplannable_one_nothing(tmp_path: Path):
    """A task with a prebuilt image and no Dockerfile gets its environment/ uploaded whole (less a compose file);
    with a Dockerfile beside the image, nothing of it reaches the sandbox; a Dockerfile platoon cannot plan, which a
    fetch refuses, gives no input files."""

    prebuilt = _task(
        tmp_path / "prebuilt",
        None,
        {"environment/data.csv": "1\n", "environment/docs/notes.md": "n\n"},
        toml='[environment]\ndocker_image = "python:3.12-slim"\n',
    )
    shadowed = _task(
        tmp_path / "shadowed",
        "FROM python:3.12-slim\nCOPY data.csv /data.csv\n",
        {"environment/data.csv": "1\n"},
        toml='[environment]\ndocker_image = "python:3.12-slim"\n',
    )
    broken = _task(tmp_path / "broken", "RUN echo no FROM\n", {"environment/data.csv": "1\n"})

    assert list(input_files(prebuilt)) == ["environment/data.csv", "environment/docs/notes.md"]
    assert input_files(shadowed) == {} and input_files(broken) == {}


def test_text_compares_as_text_and_a_binary_file_as_bytes(tmp_path: Path):
    """One table saved with CRLF, CR or LF line endings, with a byte-order mark or without, with blank lines or blanks
    at a line's end, or without its last line ended, has one digest, the SHA-256 of the table as LF text; other text
    has another.  A binary file's digest is its SHA-256, whatever its bytes; text that normalizes to nothing has an
    empty file's."""

    def digest(name: str, data: bytes) -> str:
        path = tmp_path / name
        path.write_bytes(data)
        return input_digest(path)

    table = b"PassengerId,Survived\n1,0\n2,1\n"
    saved = [
        table,
        table.replace(b"\n", b"\r\n"),
        table.replace(b"\n", b"\r"),
        b"\xef\xbb\xbf" + table.replace(b"\n", b"\r\n"),
        table.rstrip(b"\n"),
        b"\n\nPassengerId,Survived  \r\n\r\n1,0\t\r\n \n2,1 \n\n",
    ]
    assert {digest(f"saved-{n}", data) for n, data in enumerate(saved)} == {hashlib.sha256(table).hexdigest()}
    assert digest("other-row", b"PassengerId,Survived\n1,0\n2,0\n") != hashlib.sha256(table).hexdigest()
    assert digest("inner-blank", b"PassengerId, Survived\n1,0\n2,1\n") != hashlib.sha256(table).hexdigest()
    binary = b"PAR1\0" + table.replace(b"\n", b"\r\n")
    assert digest("binary", binary) == hashlib.sha256(binary).hexdigest() != digest("binary-lf", b"PAR1\0" + table)
    blank = [b"", b"\n", b" \r\n\t\n", b"\xef\xbb\xbf", b"\xef\xbb\xbf\r\n"]
    assert {"sha256:" + digest(f"blank-{n}", data) for n, data in enumerate(blank)} == {EMPTY_FILE}


def test_a_file_rewritten_with_its_size_and_modification_time_is_hashed_again(tmp_path: Path):
    """A dataset directory's digests are remembered by device, inode, size, modification time and change time: a file
    rewritten in place with its size and its modification time set back (as a copy that keeps times does) is hashed
    again, never answered with the digest of what it held."""

    dockerfile = "FROM python:3.12-slim\nRUN true\nCOPY data.csv /app/\n"
    task = _task(tmp_path / "t", dockerfile, {"environment/data.csv": "a,b\n1,2\n"})
    path = task / "environment" / "data.csv"
    stamp = path.stat().st_mtime_ns
    assert input_files(task) == {"environment/data.csv": _text("a,b\n1,2\n")}
    time.sleep(0.05)  # past the coarse clock tick that stamps a change time
    path.write_text("a,b\n3,4\n")
    os.utime(path, ns=(stamp, stamp))
    assert input_files(task) == {"environment/data.csv": _text("a,b\n3,4\n")}


def _normalized_reference(text: bytes) -> bytes:
    lines = text.replace(b"\r\n", b"\n").replace(b"\r", b"\n").split(b"\n")
    return b"".join(line + b"\n" for line in (line.rstrip(b" \t") for line in lines) if line)


def test_normalized_text_is_the_same_however_its_chunks_cut_it(monkeypatch):
    """Chunks may cut a CRLF, a line's blanks or a line too long to wait for (passed on early here, from 5 bytes):
    the text is the same as when normalized whole, with its lines split at CRLF, CR or LF, the spaces and tabs that end
    them dropped, blank ones dropped, and each one ended with LF."""

    monkeypatch.setattr(inputs, "_LONG_LINE", 5)
    rng = random.Random(20261018)
    pieces = [b"a", b"b", b",", b" ", b"\t", b"\r", b"\n", b"\r\n"]
    for _ in range(4000):
        text = b"".join(rng.choice(pieces) for _ in range(rng.randrange(0, 40)))
        cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randrange(0, 6))))
        chunks = [text[start:end] for start, end in zip([0, *cuts], [*cuts, len(text)], strict=True)]
        assert b"".join(normalized_text(chunks)) == _normalized_reference(text), (text, chunks)


def _shared_dataset(root: Path, *, game: bool = False) -> Path:
    """Twelve tasks in six groups by their index rows; tasks t0 and t6 (groups g0 and g3) hold the same table under
    other names, joining their groups.  With ``game``, every task also holds the game's code and rules."""

    tasks = {}
    for n in range(12):
        files = {"environment/instance.json": json.dumps({"name": f"t{n}"})}
        if n in (0, 6):
            files[f"environment/table-{n}.csv"] = "x,y\n1,2\n"
        if game:
            files |= {"environment/game.py": "print('the game')\n", "environment/rules.json": '{"rules": 1}'}
        copies = " ".join(sorted(path.removeprefix("environment/") for path in files))
        _task(root / f"t{n}", f"FROM python:3.12-slim\nRUN true\nWORKDIR /app\nCOPY {copies} /app/\n", files)
        tasks[f"t{n}"] = {"task_path": f"t{n}", "split_group": f"g{n // 2}"}
    (root / "task-index.json").write_text(json.dumps(tasks))
    return root


@pytest.mark.asyncio
async def test_a_lock_and_a_dataset_directory_join_the_same_groups(tmp_path: Path):
    root = _shared_dataset(tmp_path / "dataset")
    rule = {"fraction": 0.5, "group_by": "split_group"}
    locked = HarborEnvironmentService(dataset=str(root), cache=str(tmp_path / "cache"), holdout=rule)
    directory = HarborEnvironmentService(str(root), holdout=rule)

    for split in ("train", "eval"):
        assert await locked.list_tasks(split) == await directory.list_tasks(split)
    assert locked.locked is not None
    lock = read_lock(locked.locked.lock_path)
    assert lock.get("t0").input_files == {
        "environment/instance.json": _text('{"name": "t0"}'),
        "environment/table-0.csv": _text("x,y\n1,2\n"),
    }
    for service in (locked, directory):
        split = service.split_report()
        assert split is not None and split.groups is not None
        assert {"t0", "t1", "t6", "t7"} <= set(split.eval) or {"t0", "t1", "t6", "t7"} <= set(split.train)
        assert (split.groups.by_fields, split.groups.joined_groups, split.groups.groups) == (6, 2, 5)


@pytest.mark.asyncio
async def test_a_table_saved_with_other_line_endings_or_a_byte_order_mark_joins_its_copies(tmp_path: Path):
    """The same table, saved with CRLF line endings by one task and with a byte-order mark and LF line endings, the last
    one unended, by another, joins their groups as one file would, in a lock and in a dataset directory."""

    root = _shared_dataset(tmp_path / "dataset")
    (root / "t0" / "environment" / "table-0.csv").write_bytes(b"x,y\r\n1,2\r\n")
    (root / "t6" / "environment" / "table-6.csv").write_bytes(b"\xef\xbb\xbfx,y\n1,2")
    rule = {"fraction": 0.5, "group_by": "split_group"}
    locked = HarborEnvironmentService(dataset=str(root), cache=str(tmp_path / "cache"), holdout=rule)
    directory = HarborEnvironmentService(str(root), holdout=rule)

    for split in ("train", "eval"):
        assert await locked.list_tasks(split) == await directory.list_tasks(split)
    assert locked.locked is not None
    lock = read_lock(locked.locked.lock_path)
    crlf, bom = lock.get("t0").input_files, lock.get("t6").input_files
    assert crlf is not None and bom is not None
    assert crlf["environment/table-0.csv"] == bom["environment/table-6.csv"] == _text("x,y\n1,2\n")
    for service in (locked, directory):
        split = service.split_report()
        assert split is not None and split.groups is not None
        assert {"t0", "t1", "t6", "t7"} <= set(split.eval) or {"t0", "t1", "t6", "t7"} <= set(split.train)
        assert (split.groups.by_fields, split.groups.joined_groups, split.groups.groups) == (6, 2, 5)


@pytest.mark.asyncio
async def test_a_lock_written_before_input_files_answers_its_config_and_a_hold_out_asks_for_a_fetch(tmp_path: Path):
    """A lock written before platoon recorded input files still answers its config: a run without a hold-out reads
    it as it always did.  A hold-out, which would not know which tasks share a file, stops with a configuration
    error that names the fetch; a fetch, which treats such a lock as out of date, records them."""

    root = _shared_dataset(tmp_path / "dataset")
    rule = {"fraction": 0.5, "group_by": "split_group"}

    def service(**holdout: Any) -> HarborEnvironmentService:
        return HarborEnvironmentService(dataset=str(root), cache=str(tmp_path / "cache"), **holdout)

    first = service(holdout=rule)
    evaluated = await first.list_tasks("eval")
    assert first.locked is not None and first.locked.fetch is not None
    config, lock_path = first.locked.fetch, first.locked.lock_path
    lock = read_lock(lock_path)
    write_lock(replace(lock, tasks=tuple(replace(task, input_files=None) for task in lock.tasks)), lock_path)
    assert '"input_files"' not in lock_path.read_text() and not read_lock(lock_path).records_input_files

    assert answering_lock(config) is not None  # its inputs digest is unchanged: it answers the config
    assert ensure_lock(config, fetch_allowed=True).tasks == read_lock(lock_path).tasks  # read, not fetched again
    assert current_lock(config) is None  # what `platoon harbor fetch` asks: a fetch would record the input files
    assert len(await service().list_tasks("train")) == 12
    with pytest.raises(EnvSetupError, match=r"records no input files for 12 of its 12 task\(s\)") as error:
        await service(holdout=rule).list_tasks("eval")
    assert error.value.failure_kind == "env_config" and "platoon harbor fetch" in str(error.value)

    fetch(config)
    assert read_lock(lock_path).records_input_files and current_lock(config) is not None
    assert await service(holdout=rule).list_tasks("eval") == evaluated


@pytest.mark.asyncio
async def test_an_evaluation_that_shared_files_leave_empty_names_them_and_ignore_inputs(tmp_path: Path):
    """Every task holds the game's code and rules: the whole dataset is one group, which no share fits, and the
    error names a file that joined it and ``ignore_inputs``; naming the game's files there lets the rule hold out
    groups by their fields again, joined only by the table two of them share."""

    root = _shared_dataset(tmp_path / "dataset", game=True)
    rule = {"fraction": 0.25, "group_by": "split_group"}
    with pytest.raises(EnvSetupError, match="holdout: no runnable task is held out for evaluation") as error:
        await HarborEnvironmentService(str(root), holdout=rule).list_tasks("eval")
    message = str(error.value)
    assert "the largest, g0 (12 tasks), joins 6 groups by 3 file(s) such as environment/game.py" in message
    assert "name them in holdout.ignore_inputs" in message

    service = HarborEnvironmentService(
        str(root), holdout={**rule, "ignore_inputs": ["environment/game.py", "environment/rules.json"]}
    )
    evaluated = await service.list_tasks("eval")
    split = service.split_report()
    assert evaluated and split is not None and split.groups is not None
    assert (split.groups.joined_groups, split.groups.ignored_files) == (2, 2)
    record = service.split_record()
    assert record is not None
    assert record["holdout"]["ignore_inputs"] == ["environment/game.py", "environment/rules.json"]
