"""TextCraft in a container: the ``textcraft`` command of a Harbor task, and the verifier that grades it.

This one file is the whole game.  It needs Python 3.8 or later and its standard library only, and it never imports
platoon: the exporter (``python -m platoon.textcraft.harbor export``) copies it into every task as
``textcraft_game.py``, into ``environment/`` (the container gets it at ``/app/textcraft/textcraft_game.py``, which
the image's ``textcraft`` command runs) and into ``tests/`` for the verifier.  It reproduces platoon's TextCraft
environments (``platoon.textcraft.env``): the same four tools with the same arguments, the same observation text
and the same success rule.

=========================================  ===================  ===============================================
command                                    environment tool     prints
=========================================  ===================  ===============================================
``textcraft craft TARGET COUNT ING=N ...``  ``craft``            ``Successfully crafted ...`` or ``Error: ...``
``textcraft info ITEM ...``                 ``get_info``         the recipe information, as JSON
``textcraft inventory``                     ``view_inventory``   the inventory, as JSON
``textcraft finish MESSAGE``                ``finish``           the message; the episode ends
=========================================  ===================  ===============================================

``craft --json '{"ingredients": {...}, "target": [item, count]}'`` and ``info --json '[item, ...]'`` take a tool's
arguments as JSON values, the way a tool call passes them.

**State.**  ``$TEXTCRAFT_DIR`` (default: this file's directory, ``/app/textcraft`` in a task's container) holds the
task (``task.json``: goal, targets, starting inventory), the world's recipes (``recipes.json``, or
``$TEXTCRAFT_RECIPES``) and the action log (``actions.jsonl``).  The log is the state: under an exclusive lock,
every command replays the log from the starting inventory, runs, and appends its own record (``{"seq", "tool",
"args"}``, plus ``"ok"`` for a craft).  Nothing is accepted after ``finish``, or once the log no longer replays.

**Grading** (``python3 textcraft_game.py verify``, from the task's ``tests/test.sh``) replays the log with trusted
copies of the task and the recipes.  The reward is 1 when the log is an episode the rules allow that ends in
``finish`` with a message, and the inventory then holds every target count on top of the starting inventory (the
environment's rule); otherwise it is 0.  The log must be a regular file of at most 32 MiB, reached without a
symbolic link below the sandbox root.  A record that does not parse, breaks the record format or the numbering,
follows ``finish``, or claims a craft outcome the rules do not give ends the replay with reward 0: only a log the
command could have written earns a reward.  A craft the rules reject, recorded as rejected, changes nothing, as in
the environment.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import stat
import sys

VERSION = "1.0"
TASK_FORMAT = "textcraft-task/1"
RECIPES_FORMAT = "textcraft-recipes/1"
TASK_FILE = "task.json"
LOG_FILE = "actions.jsonl"
RECIPES_FILE = "recipes.json"
REWARD_FILE = "reward.json"
VERDICT_FILE = "textcraft-verdict.json"
LOG_CAP_BYTES = 32 * 1024 * 1024
CRAFTED = "Successfully crafted"
# The environment's tools and their arguments, as a log record names them.
ARGUMENTS = {"craft": ("ingredients", "target"), "get_info": ("items",), "view_inventory": (), "finish": ("message",)}
READ_ONLY = frozenset({"get_info", "view_inventory"})  # tools that change nothing
COMMANDS = {
    "craft": "craft",
    "info": "get_info",
    "get_info": "get_info",
    "inventory": "view_inventory",
    "view_inventory": "view_inventory",
    "finish": "finish",
}
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)

USAGE = """\
usage: textcraft COMMAND [ARGUMENTS]

Play the TextCraft task in $TEXTCRAFT_DIR (default /app/textcraft, the game's directory).

  textcraft craft TARGET COUNT INGREDIENT=N [INGREDIENT=N ...]
      Craft COUNT of TARGET from exactly these ingredients.  COUNT must be divisible by the recipe's
      result_count; each ingredient's N is its recipe count times COUNT / result_count.
      Example: textcraft craft m2_i2 2 m0_i1=2 m1_i1=1
  textcraft info ITEM [ITEM ...]
      Recipe information for the items, as a JSON list.
  textcraft inventory
      Your current inventory, as JSON {item_name: count}.
  textcraft finish MESSAGE
      Complete the task with a message saying what you achieved.  The task is graded on what your
      inventory gained; no command is accepted after it.

craft --json '{"ingredients": {...}, "target": [item, count]}' and info --json '[item, ...]' take the
arguments as JSON.  get_info and view_inventory are other names for info and inventory.
"""


# ---------------------------------------------------------------------------------------------------- the world


class Recipe:
    __slots__ = ("result_count", "ingredients")

    def __init__(self, result_count, ingredients):
        self.result_count = result_count
        self.ingredients = ingredients  # {item or "tag:<name>": count}, in the recipe's order


class RecipeBook:
    """A world's recipes (``recipes.json``), with the lookups of platoon's recipe databases.

    The file lists each craftable item's recipes (ingredients as ``[name, count]`` pairs, in order), the base items,
    every item's crafting depth (0 for items it does not list) and the items each ``tag:<name>`` ingredient accepts.
    """

    def __init__(self, data, sha256=None):
        if not isinstance(data, dict) or data.get("format") != RECIPES_FORMAT:
            raise ValueError("not a %s file" % RECIPES_FORMAT)
        self.name = data["name"]
        self.sha256 = sha256
        self.recipes = {
            item: [
                Recipe(recipe["result_count"], {name: count for name, count in recipe["ingredients"]})
                for recipe in recipes
            ]
            for item, recipes in data["recipes"].items()
        }
        self.base_items = frozenset(data["base_items"])
        self.depths = dict(data["depths"])
        self.tags = {tag: list(items) for tag, items in data["tags"].items()}

    @classmethod
    def load(cls, path):
        with open(path, "rb") as handle:
            raw = handle.read()
        return cls(json.loads(raw.decode("utf-8")), sha256=hashlib.sha256(raw).hexdigest())

    def get_recipes_for_item(self, item):
        return self.recipes.get(item, [])

    def can_craft(self, item):
        return item in self.recipes

    def is_base_item(self, item):
        return item in self.base_items

    def get_crafting_depth(self, item):
        return self.depths.get(item, 0)

    def get_items_for_tag(self, tag):
        return list(self.tags.get(tag, []))


def load_task(path):
    """A task file (``task.json``): its goal, targets and starting inventory, and its recipes' SHA-256."""

    with open(path, "rb") as handle:
        task = json.loads(handle.read().decode("utf-8"))
    if not isinstance(task, dict) or task.get("format") != TASK_FORMAT:
        raise ValueError("%s is not a %s file" % (path, TASK_FORMAT))
    for key in ("name", "goal", "target_items", "initial_inventory", "recipes_sha256"):
        if key not in task:
            raise ValueError("%s has no %r" % (path, key))
    return task


# ---------------------------------------------------------------------------------------------------- the game


class Game:
    """One episode: platoon's ``TextCraftEnvironment`` tools over an inventory, its rules and messages unchanged."""

    def __init__(self, book, task):
        self.book = book
        self.goal = task["goal"]
        self.target_items = dict(task["target_items"])
        self.inventory = dict(task["initial_inventory"])
        self.initial_inventory = dict(self.inventory)
        self.finished = False
        self.score = 0.0
        self.reward_misc = {}
        self.crafts = 0  # successful crafts

    def call(self, tool, args):
        """Run a tool with its arguments (the JSON values a tool call carries); the observation text it returns."""

        if tool == "craft":
            text = self.craft(args["ingredients"], args["target"])
            self.crafts += text.startswith(CRAFTED)
            return text
        if tool == "get_info":
            return json.dumps(self.get_info(args["items"]))
        if tool == "view_inventory":
            return json.dumps(dict(self.inventory))
        if tool == "finish":
            self.score, self.reward_misc = self.evaluate(args["message"])
            self.finished = True
            return args["message"]
        raise ValueError("unknown tool %r" % (tool,))

    def craft(self, ingredients, target):
        """``TextCraftEnvironment._craft``: the same checks in the same order, with the same messages."""

        if not isinstance(ingredients, dict):
            return f"Error: ingredients must be a dict, got {type(ingredients).__name__}"
        if not isinstance(target, (tuple, list)) or len(target) != 2:
            return "Error: target must be a tuple of (item_name, count)"
        target_item, target_count = target
        if not isinstance(target_item, str):
            return f"Error: target item must be a string, got {type(target_item).__name__}"
        if not isinstance(target_count, int):
            return f"Error: target count must be an int, got {type(target_count).__name__}"
        if target_count <= 0:
            return f"Error: target count must be positive, got {target_count}"
        for ing_item, ing_count in ingredients.items():
            if not isinstance(ing_item, str):
                return f"Error: ingredient name must be a string, got {type(ing_item).__name__}"
            if not isinstance(ing_count, int):
                return f"Error: ingredient count for {ing_item} must be an int, got {type(ing_count).__name__}"
            if ing_count <= 0:
                return f"Error: ingredient count for {ing_item} must be positive, got {ing_count}"

        book = self.book
        recipes = book.get_recipes_for_item(target_item)
        if not recipes:
            return f"Error: No recipe found for {target_item}"

        errors = []
        for recipe_idx, recipe in enumerate(recipes):
            if target_count % recipe.result_count != 0:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Target count {target_count} is not divisible "
                    f"by recipe result count {recipe.result_count}"
                )
                continue
            num_crafts = target_count // recipe.result_count

            ingredient_mapping = {}  # recipe ingredient (or tag) -> provided item
            recipe_error = None
            for recipe_ing, recipe_count in recipe.ingredients.items():
                total_required = recipe_count * num_crafts
                if recipe_ing.startswith("tag:"):
                    satisfying_items = book.get_items_for_tag(recipe_ing.replace("tag:", ""))
                    provided = next(
                        (
                            item
                            for item, count in ingredients.items()
                            if item in satisfying_items and count == total_required
                        ),
                        None,
                    )
                    if provided is None:
                        recipe_error = f"No ingredient provided for {recipe_ing}. Expected one of: {satisfying_items}"
                        break
                    ingredient_mapping[recipe_ing] = provided
                else:
                    if recipe_ing not in ingredients:
                        recipe_error = f"Missing required ingredient {recipe_ing}"
                        break
                    if ingredients[recipe_ing] != total_required:
                        recipe_error = (
                            f"Wrong amount of {recipe_ing}. Need {total_required}, provided {ingredients[recipe_ing]}"
                        )
                        break
                    ingredient_mapping[recipe_ing] = recipe_ing
            if recipe_error:
                errors.append(f"Recipe {recipe_idx + 1}: {recipe_error}")
                continue

            extra_ingredients = [item for item in ingredients if item not in ingredient_mapping.values()]
            if extra_ingredients:
                errors.append(
                    f"Recipe {recipe_idx + 1}: Extra ingredients not required: {', '.join(extra_ingredients)}"
                )
                continue

            missing = []
            for recipe_ing, provided_ing in ingredient_mapping.items():
                required = recipe.ingredients[recipe_ing] * num_crafts
                available = self.inventory.get(provided_ing, 0)
                if available < required:
                    missing.append(f"{provided_ing}: need {required}, have {available}")
            if missing:
                errors.append(f"Recipe {recipe_idx + 1}: Insufficient ingredients in inventory: {', '.join(missing)}")
                continue

            for recipe_ing, provided_ing in ingredient_mapping.items():
                self.inventory[provided_ing] -= recipe.ingredients[recipe_ing] * num_crafts
                if self.inventory[provided_ing] <= 0:
                    del self.inventory[provided_ing]
            crafted_count = recipe.result_count * num_crafts
            self.inventory[target_item] = self.inventory.get(target_item, 0) + crafted_count
            return f"Successfully crafted {crafted_count} {target_item}(s)"

        return f"Error: All {len(recipes)} recipe(s) failed for {target_item}. Errors:\n" + "\n".join(errors)

    def get_info(self, items):
        """``TextCraftEnvironment._get_info``."""

        book = self.book
        results = []
        for item in items:
            info = {
                "item": item,
                "can_craft": book.can_craft(item),
                "is_base": book.is_base_item(item),
                "in_inventory": self.inventory.get(item, 0),
                "crafting_depth": book.get_crafting_depth(item),
                "recipes": [],
            }
            if info["can_craft"]:
                for recipe in book.get_recipes_for_item(item):
                    expanded = {}
                    for ing, count in recipe.ingredients.items():
                        if ing.startswith("tag:"):
                            expanded[ing] = {"count": count, "use_one_of": book.get_items_for_tag(ing[4:])}
                        else:
                            expanded[ing] = count
                    info["recipes"].append({"ingredients": expanded, "result_count": recipe.result_count})
            results.append(info)
        return results

    def evaluate(self, finish_message):
        """``TextCraftEnvironment.evaluate``: success when the inventory gained every target count and the finish
        message is not empty."""

        reward_misc = {}
        score = 0.0
        if finish_message:
            if not self.target_items:
                return 0.0, {"success": False, "error": "No target items defined in task", "reward/success": 0.0}
            missing = {
                item: required - (self.inventory.get(item, 0) - self.initial_inventory.get(item, 0))
                for item, required in self.target_items.items()
                if self.inventory.get(item, 0) - self.initial_inventory.get(item, 0) < required
            }
            reward_misc["success"] = not missing
            if missing:
                reward_misc["missing_items"] = missing
            else:
                score = 1.0
            reward_misc["target_items"] = dict(self.target_items)
            reward_misc["initial_inventory"] = dict(self.initial_inventory)
            reward_misc["final_inventory"] = dict(self.inventory)
        else:
            reward_misc["success"] = False
            reward_misc["error"] = "Episode ended without calling finish() - likely timeout or max steps reached"
        reward_misc["reward/success"] = score
        return score, reward_misc


# ---------------------------------------------------------------------------------------------------- the log


class LogError(ValueError):
    """An action log the game could not have written: its replay stops there."""


def _unique_pairs(pairs):
    found = {}
    for key, value in pairs:
        if key in found:
            raise ValueError("duplicate key %r" % (key,))
        found[key] = value
    return found


def _no_constant(name):
    raise ValueError("%s is not a JSON number" % name)


def _finite_float(text):
    value = float(text)
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError("%s is out of range" % text)
    return value


def parse_json(text):
    """Strict JSON: no duplicate keys, no NaN or Infinity (also none that overflows, such as 1e999), so every value
    it returns can be written back to the log."""

    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_no_constant, parse_float=_finite_float)


def encode_record(seq, tool, args, ok=None):
    """A log line: ``{"seq", "tool", "args"}``, plus ``"ok"`` (the craft succeeded) for a craft."""

    record = {"seq": seq, "tool": tool, "args": args}
    if tool == "craft":
        record["ok"] = bool(ok)
    return (json.dumps(record, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def check_arguments(tool, args):
    """``args`` as a log record or a command may give them to ``tool``; ValueError says what is wrong."""

    if not isinstance(args, dict) or sorted(args) != sorted(ARGUMENTS[tool]):
        raise ValueError("%s takes the arguments %s" % (tool, ", ".join(ARGUMENTS[tool]) or "none"))
    if tool == "get_info" and (
        not isinstance(args["items"], list) or not all(isinstance(item, str) for item in args["items"])
    ):
        raise ValueError("get_info takes a list of item names")
    if tool == "finish" and not isinstance(args["message"], str):
        raise ValueError("finish takes a message string")


def _record(line, seq):
    try:
        record = parse_json(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise LogError("record %d is not a JSON object: %s" % (seq, exc)) from None
    if not isinstance(record, dict):
        raise LogError("record %d is not a JSON object" % seq)
    tool = record.get("tool")
    if not isinstance(tool, str) or tool not in ARGUMENTS:
        raise LogError("record %d names no tool of the game: %r" % (seq, tool))
    keys = {"seq", "tool", "args", "ok"} if tool == "craft" else {"seq", "tool", "args"}
    if set(record) != keys:
        raise LogError("record %d must have exactly the keys %s" % (seq, sorted(keys)))
    if type(record["seq"]) is not int or record["seq"] != seq:
        raise LogError("record %d is numbered %r" % (seq, record["seq"]))
    try:
        check_arguments(tool, record["args"])
    except ValueError as exc:
        raise LogError("record %d: %s" % (seq, exc)) from None
    if tool == "craft" and not isinstance(record["ok"], bool):
        raise LogError("record %d: ok must be true or false" % seq)
    return tool, record["args"], record.get("ok")


def replay(game, data):
    """Apply a log's records to ``game``; how many there were.  LogError at the first record the game could not
    have written: the replay stops there.

    The records are read one at a time, and those of ``get_info`` and ``view_inventory``, which change nothing, have
    their arguments checked but their output never rebuilt, so replaying any log within the size cap takes seconds
    and a bounded amount of memory (a ``get_info`` record can name millions of items)."""

    if not data:
        return 0
    if not data.endswith(b"\n"):
        raise LogError("the last record does not end with a newline")
    seq = start = 0
    while start < len(data):
        end = data.index(b"\n", start)
        seq += 1
        if game.finished:
            raise LogError("record %d follows finish" % seq)
        tool, args, ok = _record(data[start:end], seq)
        start = end + 1
        if tool in READ_ONLY:
            continue
        try:
            text = game.call(tool, args)
        except Exception as exc:  # arguments no command writes
            raise LogError("record %d cannot be applied: %s: %s" % (seq, type(exc).__name__, exc)) from None
        if tool == "craft" and text.startswith(CRAFTED) != ok:
            claim = "succeeded" if ok else "failed"
            raise LogError("record %d says the craft %s, but the rules give: %s" % (seq, claim, text))
    return seq


# ---------------------------------------------------------------------------------------------------- the command


class UsageError(ValueError):
    pass


class Result:
    """What a command printed and its exit status: 0 done, 1 a game error (a rejected craft, the episode is over),
    2 a usage error (nothing was run or logged), 3 the task cannot be played."""

    def __init__(self, code, out="", err=""):
        self.code = code
        self.out = out
        self.err = err


def parse_command(argv):
    """``(tool, args)`` of a command line; UsageError when it is not one."""

    if not argv:
        raise UsageError("no command")
    tool = COMMANDS.get(argv[0])
    if tool is None:
        raise UsageError("unknown command %r" % argv[0])
    rest = argv[1:]
    if tool == "craft":
        if rest[:1] == ["--json"]:
            if len(rest) != 2:
                raise UsageError("craft --json takes one JSON object")
            args = _json_argument(rest[1])
            if not isinstance(args, dict) or sorted(args) != ["ingredients", "target"]:
                raise UsageError('craft --json takes {"ingredients": {...}, "target": [item, count]}')
            return tool, args
        if len(rest) < 2:
            raise UsageError("craft takes TARGET COUNT INGREDIENT=N ...")
        ingredients = {}
        for word in rest[2:]:
            name, sep, count = word.rpartition("=")
            if not sep or not name:
                raise UsageError("ingredients are INGREDIENT=N, got %r" % word)
            if name in ingredients:
                raise UsageError("ingredient %r is given twice" % name)
            ingredients[name] = _integer(count, "the count of %s" % name)
        return tool, {"ingredients": ingredients, "target": [rest[0], _integer(rest[1], "COUNT")]}
    if tool == "get_info":
        if rest[:1] == ["--json"]:
            if len(rest) != 2:
                raise UsageError("info --json takes one JSON list")
            items = _json_argument(rest[1])
        else:
            items = list(rest)
        args = {"items": items}
    elif tool == "view_inventory":
        if rest:
            raise UsageError("inventory takes no arguments")
        args = {}
    else:
        if not rest:
            raise UsageError("finish takes a MESSAGE saying what you achieved")
        args = {"message": " ".join(rest)}
    try:
        check_arguments(tool, args)
    except ValueError as exc:
        raise UsageError(str(exc)) from None
    return tool, args


def _json_argument(text):
    try:
        return parse_json(text)
    except (ValueError, RecursionError) as exc:
        raise UsageError("not valid JSON: %s" % exc) from None


def _integer(text, what):
    try:
        return int(text)
    except ValueError:
        raise UsageError("%s must be an integer, got %r" % (what, text)) from None


def default_directory():
    return os.environ.get("TEXTCRAFT_DIR") or os.path.dirname(os.path.abspath(__file__))


def run(argv, directory=None, recipes=None):
    """Run a command line against the task in ``directory`` (default ``$TEXTCRAFT_DIR``, else this file's
    directory), with the recipes ``recipes`` (default ``$TEXTCRAFT_RECIPES``, else the directory's
    ``recipes.json``)."""

    if not argv or argv[0] in ("-h", "--help", "help"):
        return Result(0 if argv else 2, USAGE)
    if argv[0] == "--version":
        return Result(0, "textcraft %s" % VERSION)
    try:
        tool, args = parse_command(argv)
    except UsageError as exc:
        return Result(2, err="textcraft: %s\n\n%s" % (exc, USAGE))
    directory = directory or default_directory()
    try:
        task = load_task(os.path.join(directory, TASK_FILE))
        book = RecipeBook.load(recipes or os.environ.get("TEXTCRAFT_RECIPES") or os.path.join(directory, RECIPES_FILE))
    except (OSError, ValueError, KeyError) as exc:
        return Result(3, err="textcraft: cannot load the task: %s" % exc)
    if task["recipes_sha256"] != book.sha256:
        return Result(3, err="textcraft: the recipes (%s) are not the ones of this task" % book.sha256)
    path = os.path.join(directory, LOG_FILE)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | _CLOEXEC, 0o644)
    except OSError as exc:
        why = "it is a symbolic link" if exc.errno == errno.ELOOP else exc.strerror
        return Result(3, err="textcraft: cannot open the action log %s: %s" % (path, why))
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError:
            pass  # a file system without locks: commands are not expected to overlap
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return Result(3, err="textcraft: the action log %s is not a regular file" % path)
        data = _read_all(fd)
        game = Game(book, task)
        try:
            seq = replay(game, data) + 1
        except LogError as exc:
            return Result(
                3, err="textcraft: the action log %s does not replay (%s); the episode cannot continue" % (path, exc)
            )
        if game.finished:
            return Result(
                1, "Error: the episode is finished (textcraft finish was called); no command is accepted after it"
            )
        try:
            text = game.call(tool, args)
        except Exception as exc:
            return Result(2, err="textcraft: cannot run %s: %s: %s" % (tool, type(exc).__name__, exc))
        ok = text.startswith(CRAFTED)
        line = encode_record(seq, tool, args, ok)
        if len(data) + len(line) > LOG_CAP_BYTES:
            return Result(1, "Error: the action log is full (%d bytes); no command is accepted" % LOG_CAP_BYTES)
        _write_all(fd, line)
    finally:
        os.close(fd)
    return Result(1 if tool == "craft" and not ok else 0, text)


def _read_all(fd):
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


# ---------------------------------------------------------------------------------------------------- the verifier


def read_untrusted(root, relative, cap=LOG_CAP_BYTES):
    """The bytes of ``root/relative``, reached without following a symbolic link below ``root``; None when it does
    not exist.  LogError when it is not a regular file, or holds more than ``cap`` bytes."""

    parts = [part for part in relative.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError("%r is not a path below the root" % relative)
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | _CLOEXEC)
    try:
        for part in parts[:-1]:
            try:
                below = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | _CLOEXEC, dir_fd=directory)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise LogError("%s on the way to the action log is not a directory (%s)" % (part, exc.strerror))
            os.close(directory)
            directory = below
        try:
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | _CLOEXEC, dir_fd=directory)
        except FileNotFoundError:
            return None
        except OSError as exc:
            why = "it is a symbolic link" if exc.errno == errno.ELOOP else exc.strerror
            raise LogError("cannot open the action log: %s" % why)
    finally:
        os.close(directory)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise LogError("the action log is not a regular file")
        if info.st_size > cap:
            raise LogError("the action log holds %d bytes; the limit is %d" % (info.st_size, cap))
        chunks, total = [], 0
        while total <= cap:
            chunk = os.read(fd, min(1 << 20, cap + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > cap:
            raise LogError("the action log holds more than %d bytes" % cap)
        return b"".join(chunks)
    finally:
        os.close(fd)


def verify(task_path, recipes_path, root, log, out=None):
    """Grade the log at ``root/log`` against the trusted task and recipes; the verdict (``reward`` and why).
    Writes ``reward.json`` and ``textcraft-verdict.json`` into ``out`` when given."""

    task = load_task(task_path)
    book = RecipeBook.load(recipes_path)
    if task["recipes_sha256"] != book.sha256:
        raise ValueError("the verifier's recipes %s are not the task's %s" % (book.sha256, task["recipes_sha256"]))
    game = Game(book, task)
    verdict = {"reward": 0.0, "task": task["name"], "version": VERSION, "records": 0, "crafts": 0, "finished": False}
    try:
        data = read_untrusted(root, log)
        if data is None:
            verdict["reason"] = "no action log: no textcraft command ran"
        else:
            verdict["log_bytes"] = len(data)
            verdict["records"] = replay(game, data)
            verdict["crafts"] = game.crafts
            verdict["finished"] = game.finished
            if not game.finished:
                verdict["reason"] = "the episode did not end with textcraft finish"
            elif game.score >= 1.0:
                verdict["reward"] = 1.0
                verdict["reason"] = "finished with every target crafted"
            elif game.reward_misc.get("missing_items"):
                verdict["reason"] = "finished without every target crafted"
                verdict["missing_items"] = game.reward_misc["missing_items"]
            else:
                verdict["reason"] = "finished with an empty message"
    except LogError as exc:
        verdict["reason"] = "the action log is not one the game wrote: %s" % exc
    if out is not None:
        _write_new(out, VERDICT_FILE, json.dumps(verdict, indent=1, sort_keys=True) + "\n")
        _write_new(out, REWARD_FILE, json.dumps({"reward": verdict["reward"]}) + "\n")
    return verdict


def _write_new(directory, name, text):
    """Write ``directory/name`` through a new file and a rename, so the name ends up a regular file whatever was
    there."""

    temporary = os.path.join(directory, ".%s.%d.tmp" % (name, os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | _CLOEXEC, 0o644)
    try:
        _write_all(fd, text.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    os.replace(temporary, os.path.join(directory, name))


def verify_main(argv):
    parser = argparse.ArgumentParser(prog="textcraft_game.py verify", description="Grade a TextCraft action log.")
    parser.add_argument("--task", required=True, help="the trusted task.json")
    parser.add_argument("--recipes", required=True, help="the trusted recipes.json")
    parser.add_argument("--root", required=True, help="the sandbox root the log path is below")
    parser.add_argument("--log", required=True, help="the log's path below the root")
    parser.add_argument("--out", required=True, help="the verifier's output directory (reward.json)")
    options = parser.parse_args(argv)
    try:
        verdict = verify(options.task, options.recipes, options.root, options.log, options.out)
    except (OSError, ValueError, KeyError) as exc:
        print("textcraft verify: cannot grade: %s" % exc, file=sys.stderr)
        return 1
    print(
        "reward %g: %s (%d records, %d crafts)"
        % (verdict["reward"], verdict["reason"], verdict["records"], verdict["crafts"])
    )
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["verify"]:
        return verify_main(argv[1:])
    result = run(argv)
    if result.out:
        print(result.out)
    if result.err:
        print(result.err, file=sys.stderr)
    return result.code


if __name__ == "__main__":
    sys.exit(main())
