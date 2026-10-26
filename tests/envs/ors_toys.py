"""Toy ORS environments for the ORS service tests.

They use only the API that openreward 0.1.125 and 0.1.163 share: ``@tool`` with
one pydantic parameter (or none), ``ToolOutput(finished=True, reward=...)``,
and the ``list_splits``/``list_tasks``/``get_prompt`` methods; only
``FlaggedEnv``, defined where the SDK has them (0.1.163), uses its tool flags
and ``@terminal``. Tests reference them as ``python://ors_toys:<Class>``
(pytest puts this directory on the path).
"""

from __future__ import annotations

import asyncio
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path

import openreward.environments as _sdk
from openreward.environments import Environment, ImageBlock, TextBlock, ToolOutput, tool
from pydantic import BaseModel, ConfigDict, Field

EVENTS: list[str] = []  # setup/teardown calls, for lifecycle assertions


class AddParams(BaseModel):
    amount: int = Field(description="How much to add to the counter.")


class SubmitParams(BaseModel):
    answer: int


class QuizAnswer(BaseModel):
    answer: str


class HandIn(BaseModel):
    answer: int
    note: str = ""


class SearchParams(BaseModel):
    model_config = ConfigDict(json_schema_extra={"x-openhands-tool-routing": {"lane": "search"}})

    query: str
    title: str = ""  # a property named "title" must survive title stripping


class CounterEnv(Environment):
    """Reach the task's target with ``add``, then ``submit`` it."""

    platoon_terminal_tools = ("submit",)
    platoon_read_only_tools = ("peek", "search")
    platoon_tool_meta = {"add": {"example.dev/kind": "mutation"}}

    def __init__(self, task_spec: dict = {}, secrets: dict = {}) -> None:  # noqa: B006 - the SDK's signature
        super().__init__(task_spec, secrets)
        self.total = 0
        self.secret = secrets.get("token")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": f"{split}-{index}", "target": index + 2} for index in range(3)]

    def setup(self) -> None:
        EVENTS.append(f"setup:{type(self).__name__}")

    def teardown(self) -> None:
        EVENTS.append(f"teardown:{type(self).__name__}")

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=f"Reach {self.task_spec['target']}.")]

    @tool
    async def add(self, params: AddParams) -> ToolOutput:
        """Add to the counter."""
        self.total += params.amount
        return ToolOutput(blocks=[TextBlock(text=f"total={self.total}")], metadata={"total": self.total})

    @tool
    def peek(self) -> ToolOutput:
        """Show the counter and a picture of it."""
        return ToolOutput(blocks=[TextBlock(text=str(self.total)), ImageBlock(data="aGk=", mimeType="image/png")])

    @tool
    async def search(self, params: SearchParams) -> ToolOutput:
        """Search the docs."""
        return ToolOutput(blocks=[TextBlock(text=f"no results for {params.query}")])

    search._meta = {"provider.dev/source": "docs"}  # type: ignore[attr-defined]

    @tool
    async def submit(self, params: SubmitParams) -> ToolOutput:
        """Submit the final total; -1 forfeits (an invalid grade)."""
        if params.answer == -1:
            return ToolOutput(blocks=[TextBlock(text="forfeit")], metadata={"invalid": True}, finished=True)
        correct = params.answer == self.total == self.task_spec["target"]
        return ToolOutput(
            blocks=[TextBlock(text="correct" if correct else "wrong")],
            reward=1.0 if correct else 0.0,
            metadata={"answer": params.answer},
            finished=True,
        )

    @tool
    async def give_up(self) -> ToolOutput:
        """End the episode without an answer."""
        return ToolOutput(blocks=[TextBlock(text="bye")], finished=True)

    @tool
    async def flag_error(self) -> ToolOutput:
        """Report an env-declared tool error."""
        return ToolOutput(
            blocks=[TextBlock(text="bad state")], metadata={"openhands.dev/tool-error": {"message": "try again"}}
        )

    @tool
    async def explode(self) -> ToolOutput:
        """Raise inside the env."""
        raise RuntimeError("boom")


STALE_COMPLETION_LINE = (
    "If you believe the task is completed, you can either call the `local-claim_done` tool or respond without "
    "calling any tool to indicate completion."
)


class StalePromptEnv(CounterEnv):
    """A counter whose prompt still has an old Toolathlon image's stale completion line."""

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=f"Reach {self.task_spec['target']}.\n{STALE_COMPLETION_LINE}\nThen submit.")]


def drop_stale_completion_line(prompt: str) -> str:
    """A ``prompt_filter`` that drops the stale completion line."""

    return "".join(line for line in prompt.splitlines(keepends=True) if line.strip() != STALE_COMPLETION_LINE)


class GradedCounterEnv(CounterEnv):
    """A counter that grades its state when nothing was submitted."""

    def platoon_grade(self) -> dict:
        return {"reward": self.total / 10, "metadata": {"total": self.total}}


class BrokenSetupEnv(CounterEnv):
    def setup(self) -> None:
        raise RuntimeError("no capacity")


class SlowSetupEnv(CounterEnv):
    """A sync setup that takes a second to acquire a resource; teardown releases whatever it finds."""

    def setup(self) -> None:
        time.sleep(1.0)
        self.resource = "acquired"
        EVENTS.append("setup:done")

    def teardown(self) -> None:
        EVENTS.append(f"teardown:{getattr(self, 'resource', None)}")


class AsyncSlowSetupEnv(CounterEnv):
    """An async setup that takes long enough to be cancelled."""

    async def setup(self) -> None:
        await asyncio.sleep(5)
        EVENTS.append("setup:done")

    def teardown(self) -> None:
        EVENTS.append("teardown:async")


class SlowToolEnv(CounterEnv):
    """A sync tool that takes a second."""

    @tool
    def slow(self) -> ToolOutput:
        """Take a second."""
        time.sleep(1.0)
        EVENTS.append("slow:done")
        return ToolOutput(blocks=[TextBlock(text="slow")])


HELD = threading.Event()  # set once HeldToolEnv's tool runs
RELEASE = threading.Event()  # lets HeldToolEnv's tool return


class HeldToolEnv(CounterEnv):
    """A sync tool that sets ``HELD`` and runs until a test sets ``RELEASE``."""

    @tool
    def held(self) -> ToolOutput:
        """Wait to be released."""
        HELD.set()
        RELEASE.wait(30)
        EVENTS.append("held:done")
        return ToolOutput(blocks=[TextBlock(text="released")])


class SilentRaiseEnv(CounterEnv):
    """Tools that raise exceptions whose message is empty."""

    @tool
    async def wait_inner(self) -> ToolOutput:
        """Wait on an inner call that times out."""
        await asyncio.wait_for(asyncio.sleep(10), 0.01)  # TimeoutError()
        return ToolOutput(blocks=[TextBlock(text="unreachable")])

    @tool
    async def check(self) -> ToolOutput:
        """Check an invariant that does not hold."""
        assert self.total < 0  # AssertionError()
        return ToolOutput(blocks=[TextBlock(text="unreachable")])


class BackendFailureEnv(CounterEnv):
    """Tools that fail the way an env's own backing service fails: with the openreward SDK's session and transport
    errors (which both supported SDKs name and word alike), as a sandbox or an API client the env uses raises them."""

    @tool
    async def lose_sandbox(self) -> ToolOutput:
        """Run a command in a sandbox that is gone."""
        from openreward.api._session.session import SessionTerminatedError

        raise SessionTerminatedError("server returned 410: Gone", sid="sandbox-1", kind="sandbox")

    @tool
    async def exhaust_retries(self) -> ToolOutput:
        """Call a backing API that keeps failing."""
        from openreward.api._session.http import MaxRetriesError

        raise MaxRetriesError("Exceeded 5 retries for /exec", errors=[ConnectionResetError("reset by peer")])

    @tool
    async def fetch_page(self) -> ToolOutput:
        """Fetch a page that cannot be reached: httpx's ConnectError, a TransportError that is not the SDK's."""
        import httpx

        raise httpx.ConnectError("All connection attempts failed")


class RecountEnv(CounterEnv):
    """A counter whose ``add`` is described otherwise: the same tool names as ``CounterEnv``, another listing."""

    @tool
    async def add(self, params: AddParams) -> ToolOutput:
        """Add to the counter, then count again."""
        self.total += params.amount
        return ToolOutput(blocks=[TextBlock(text=f"total={self.total}")], metadata={"total": self.total})


if hasattr(_sdk, "terminal"):  # openreward 0.1.163: the SDK's own tool flags and @terminal

    class Verdict(BaseModel):
        verdict: str = Field(description="The answer, in one word.")

    class FlaggedEnv(CounterEnv):
        """The SDK's own hints, which an env server does not send: an exclusive hand-in, a concurrent lookup and a
        ``@terminal`` tool whose parameter has a description."""

        platoon_terminal_tools = ()

        @_sdk.tool(exclusive=True)
        async def hand_in(self, params: HandIn) -> ToolOutput:
            """Hand in an answer with a note; this ends the episode."""
            reward = float(params.answer == self.total == self.task_spec["target"])
            return ToolOutput(blocks=[TextBlock(text="handed in")], reward=reward, finished=True)

        @_sdk.tool(concurrent=True)
        async def lookup(self, params: SearchParams) -> ToolOutput:
            """Look something up; safe to run while other calls run."""
            return ToolOutput(blocks=[TextBlock(text=f"nothing on {params.query}")])

        @_sdk.terminal
        @_sdk.tool
        async def respond(self, params: Verdict) -> ToolOutput:
            """Answer in one word."""
            return ToolOutput(blocks=[TextBlock(text="ok")], reward=float(params.verdict == "four"), finished=True)


class TallyEnv(Environment):
    """Tallies under its own asyncio.Lock held across an await, and fills a table from a sync tool.

    It has no teardown, so a copy fork deep-copies it.
    """

    def __init__(self, task_spec: dict = {}, secrets: dict = {}) -> None:  # noqa: B006 - the SDK's signature
        super().__init__(task_spec, secrets)
        self.lock = asyncio.Lock()
        self.total = 0
        self.added: list[int] = []
        self.table: dict[int, int] = {}

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"name": "tally"}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Tally.")]

    @tool
    async def slow_add(self, params: AddParams) -> ToolOutput:
        """Add, holding the lock across an await."""
        async with self.lock:
            self.total += params.amount
            await asyncio.sleep(0.3)
            self.added.append(params.amount)
        return ToolOutput(blocks=[TextBlock(text=f"total={self.total}")])

    @tool
    def fill(self, params: AddParams) -> ToolOutput:
        """Fill the table, slowly, from a thread."""
        for index in range(params.amount):
            self.table[index] = index
            if index % 100 == 0:
                time.sleep(0.001)
        return ToolOutput(blocks=[TextBlock(text=f"rows={len(self.table)}")])


class WorkdirEnv(Environment):
    """Keeps its answer in a workdir it creates in setup and removes in teardown: a resource a deep copy shares."""

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"answer": "42"}]

    def setup(self) -> None:
        self.workdir = Path(tempfile.mkdtemp(prefix="ors-workdir-"))

    def teardown(self) -> None:
        shutil.rmtree(self.workdir)
        EVENTS.append(f"teardown:{self.workdir.name}")

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text="Write the answer, then submit.")]

    @tool
    async def write(self, params: QuizAnswer) -> ToolOutput:
        """Write an answer to the workdir."""
        (self.workdir / "answer.txt").write_text(params.answer)
        return ToolOutput(blocks=[TextBlock(text="written")])

    @tool
    async def submit(self) -> ToolOutput:
        """Grade the written answer."""
        answer = (self.workdir / "answer.txt").read_text() if (self.workdir / "answer.txt").exists() else None
        return ToolOutput(blocks=[TextBlock(text="graded")], reward=float(answer == "42"), finished=True)


class CopyableWorkdirEnv(WorkdirEnv):
    """A workdir env that copies its workdir for a copy fork."""

    def platoon_copy(self) -> CopyableWorkdirEnv:
        clone = CopyableWorkdirEnv(self.task_spec)
        clone.workdir = Path(tempfile.mkdtemp(prefix="ors-workdir-"))
        shutil.copytree(self.workdir, clone.workdir, dirs_exist_ok=True)
        return clone


class CraftParams(BaseModel):
    item: str


class CraftEnv(Environment):
    """TextCraft-like: crafted items land in an inventory that ``platoon_fork`` children share."""

    def __init__(self, task_spec: dict = {}, secrets: dict = {}, inventory: dict | None = None) -> None:  # noqa: B006
        super().__init__(task_spec, secrets)
        self.inventory: dict[str, int] = inventory if inventory is not None else {}
        self.start = dict(self.inventory)

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[dict]:
        return [{"goal": "craft 3 plank", "item": "plank", "count": 3}]

    def get_prompt(self) -> list[TextBlock]:
        return [TextBlock(text=self.task_spec["goal"])]

    @tool
    async def craft(self, params: CraftParams) -> ToolOutput:
        """Craft one item."""
        self.inventory[params.item] = self.inventory.get(params.item, 0) + 1
        return ToolOutput(blocks=[TextBlock(text=f"{params.item}: {self.inventory[params.item]}")])

    @tool
    async def done(self) -> ToolOutput:
        """Finish: rewarded when this agent crafted its goal's items."""
        item, count = self.task_spec["item"], self.task_spec["count"]
        crafted = self.inventory.get(item, 0) - self.start.get(item, 0)
        return ToolOutput(blocks=[TextBlock(text=f"crafted {crafted}")], reward=float(crafted >= count), finished=True)

    def platoon_fork(self, task_spec: dict) -> CraftEnv:
        match = re.fullmatch(r"craft (\d+) (\w+)", task_spec["goal"])
        assert match, task_spec
        spec = {**task_spec, "item": match[2], "count": int(match[1])}
        return CraftEnv(spec, inventory=self.inventory)


def shelf_envs() -> tuple[type, type]:
    """A fresh parent class and subclass with different tasks on the same split.

    Fresh classes each call, since the SDK caches task lists on the class: a test that wants a particular class
    to list first needs classes that have never listed.
    """

    class ShelfEnv(Environment):
        """Three books on the train shelf."""

        @classmethod
        def list_splits(cls) -> list[str]:
            return ["train"]

        @classmethod
        def list_tasks(cls, split: str) -> list[dict]:
            return [{"name": f"book-{index}"} for index in range(3)]

        def get_prompt(self) -> list[TextBlock]:
            return [TextBlock(text=f"Read {self.task_spec['name']}.")]

        @tool
        async def read(self) -> ToolOutput:
            """Read the book."""
            return ToolOutput(blocks=[TextBlock(text="read")], reward=1.0, finished=True)

    class TallShelfEnv(ShelfEnv):
        """Five books on the same split: a task cache shared with ShelfEnv would hide two of them."""

        @classmethod
        def list_tasks(cls, split: str) -> list[dict]:
            return [{"name": f"tall-{index}"} for index in range(5)]

    return ShelfEnv, TallShelfEnv


class TrackedCraftEnv(CraftEnv):
    """A CraftEnv whose teardown is recorded, so a test can count how often the shared env is torn down."""

    def teardown(self) -> None:
        EVENTS.append("teardown:TrackedCraftEnv")


class SlowForkCraftEnv(TrackedCraftEnv):
    """A tracked CraftEnv whose sync platoon_fork takes half a second; a goal of "nothing" gets no env."""

    def platoon_fork(self, task_spec: dict) -> SlowForkCraftEnv | None:
        time.sleep(0.5)
        EVENTS.append("fork:done")
        if task_spec["goal"] == "nothing":
            return None
        match = re.fullmatch(r"craft (\d+) (\w+)", task_spec["goal"])
        assert match, task_spec
        return SlowForkCraftEnv({**task_spec, "item": match[2], "count": int(match[1])}, inventory=self.inventory)
