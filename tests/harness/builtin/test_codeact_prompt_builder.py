"""CodeAct prompts: the sequence-extension property, the two prompt modes, the original task text and reply parse.

The key property is that for sequence_extension mode, each step's observation should be
a prefix of the next step's observation, enabling tinker to merge consecutive steps
into a single Datum.
"""

import pytest

from platoon.harness.builtin.codeact import CellFunction, CodeActPromptBuilder, CodeActState, CodeActStep, Parameter
from platoon.task import Task


class SimplePromptBuilder(CodeActPromptBuilder):
    """Simple prompt builder with hardcoded system prompt (no Jinja templates)."""

    def build_system_prompt(self, state: CodeActState, **context) -> str:
        return "You are a helpful assistant."


def create_test_observation(history_length: int) -> CodeActState:
    """Create a test prompt state with the given history length."""
    history = []
    for i in range(history_length):
        step = CodeActStep(
            thought=f"Thinking about step {i}",
            code=f"action_{i}()",
            output=f"Result of step {i}",
        )
        history.append(step)

    return CodeActState(
        task=str(Task(id="test_task", goal="Solve the test problem")),
        action_space="You can call action_X() to perform action X.",
        history=history,
    )


def messages_to_text(messages: list) -> str:
    """Convert messages to a text representation for prefix checking."""
    parts = []
    for msg in messages:
        parts.append(f"[{msg['role']}]\n{msg['content']}")
    return "\n\n".join(parts)


def assert_is_char_prefix(shorter: str, longer: str, msg: str = ""):
    """Assert that shorter is a character-level prefix of longer.

    Provides detailed error message showing exactly where the mismatch occurs.
    """
    assert len(shorter) <= len(longer), (
        f"{msg}: shorter string ({len(shorter)} chars) is longer than longer string ({len(longer)} chars)"
    )

    # Check character by character
    for i, (c1, c2) in enumerate(zip(shorter, longer)):
        if c1 != c2:
            # Show context around the mismatch
            start = max(0, i - 20)
            end = min(len(shorter), i + 20)
            raise AssertionError(
                f"{msg}: Character mismatch at position {i}.\n"
                f"  Expected: ...{repr(shorter[start:end])}...\n"
                f"  Got:      ...{repr(longer[start:end])}...\n"
                f"  Char {i}: {repr(c1)} != {repr(c2)}"
            )

    # Verify the prefix matches exactly
    assert longer[: len(shorter)] == shorter, (
        f"{msg}: Prefix mismatch. First {len(shorter)} chars should match exactly."
    )


class TestSequenceExtension:
    """Test that sequence_extension prompts are prefixes of subsequent prompts."""

    def test_sequence_extension_is_prefix_char_level(self):
        """Each step's prompt should be a character-level prefix of the next step's prompt."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        # Create observations with increasing history
        obs_0 = create_test_observation(0)
        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)
        obs_3 = create_test_observation(3)

        # Build prompts
        msgs_0 = builder.build_messages(obs_0)
        msgs_1 = builder.build_messages(obs_1)
        msgs_2 = builder.build_messages(obs_2)
        msgs_3 = builder.build_messages(obs_3)

        # Convert to text for comparison
        text_0 = messages_to_text(msgs_0)
        text_1 = messages_to_text(msgs_1)
        text_2 = messages_to_text(msgs_2)
        text_3 = messages_to_text(msgs_3)

        # Check character-level prefix property
        assert_is_char_prefix(text_0, text_1, "Step 0 should be char prefix of Step 1")
        assert_is_char_prefix(text_1, text_2, "Step 1 should be char prefix of Step 2")
        assert_is_char_prefix(text_2, text_3, "Step 2 should be char prefix of Step 3")

        # Also verify the lengths are strictly increasing
        assert len(text_0) < len(text_1) < len(text_2) < len(text_3), (
            f"Prompt lengths should strictly increase: {len(text_0)} < {len(text_1)} < {len(text_2)} < {len(text_3)}"
        )

    def test_sequence_extension_exact_prefix_bytes(self):
        """Verify that the prefix matches exactly byte-for-byte."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)

        text_1 = messages_to_text(builder.build_messages(obs_1))
        text_2 = messages_to_text(builder.build_messages(obs_2))

        # The first len(text_1) characters of text_2 should be identical to text_1
        prefix_of_text_2 = text_2[: len(text_1)]
        assert prefix_of_text_2 == text_1, (
            f"First {len(text_1)} chars of step 2 should exactly equal step 1.\n"
            f"Difference starts at char {next((i for i, (a, b) in enumerate(zip(text_1, prefix_of_text_2)) if a != b), len(text_1))}"  # noqa: E501
        )

    def test_no_sequence_extension_not_prefix(self):
        """no_sequence_extension prompts should NOT be prefixes (they rebuild history)."""
        builder = SimplePromptBuilder(prompt_mode="no_sequence_extension")

        obs_0 = create_test_observation(0)
        obs_1 = create_test_observation(1)

        msgs_0 = builder.build_messages(obs_0)
        msgs_1 = builder.build_messages(obs_1)

        text_0 = messages_to_text(msgs_0)
        text_1 = messages_to_text(msgs_1)

        # no_sequence_extension mode rebuilds the entire history in the user message
        # So they should NOT be prefixes (the "Action History" section changes)
        assert not text_1.startswith(text_0), "no_sequence_extension prompts should not be prefixes"

    def test_sequence_extension_grows_by_appending(self):
        """sequence_extension prompts should grow by appending, not rebuilding."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)

        msgs_1 = builder.build_messages(obs_1)
        msgs_2 = builder.build_messages(obs_2)

        # Step 2 should have more messages than step 1
        assert len(msgs_2) > len(msgs_1), "More history should mean more messages"

        # The first messages should be identical
        for i in range(len(msgs_1)):
            assert msgs_1[i] == msgs_2[i], f"Message {i} should be identical"

    def test_sequence_extension_message_structure(self):
        """Verify the sequence_extension message structure."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension")

        obs_2 = create_test_observation(2)
        msgs = builder.build_messages(obs_2)

        # Expected structure:
        # [0] system
        # [1] user (initial task)
        # [2] assistant (action 0)
        # [3] user (output 0)
        # [4] assistant (action 1)
        # [5] user (output 1)

        assert len(msgs) == 6, f"Expected 6 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[4]["role"] == "assistant"
        assert msgs[5]["role"] == "user"

    def test_no_sequence_extension_message_structure(self):
        """Verify the no_sequence_extension message structure."""
        builder = SimplePromptBuilder(prompt_mode="no_sequence_extension")

        obs_2 = create_test_observation(2)
        msgs = builder.build_messages(obs_2)

        # no_sequence_extension always has exactly 2 messages
        assert len(msgs) == 2, f"Expected 2 messages, got {len(msgs)}"
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        # The user message contains all history
        assert "action_0" in msgs[1]["content"]
        assert "action_1" in msgs[1]["content"]


class TestPromptModeDefault:
    """Test that sequence_extension is the default prompt mode."""

    def test_default_is_sequence_extension(self):
        """CodeActPromptBuilder should default to sequence_extension."""
        builder = CodeActPromptBuilder()
        assert builder.prompt_mode == "sequence_extension"


class TestIncludeReasoning:
    """Test the include_reasoning flag."""

    def test_default_is_include_reasoning(self):
        """CodeActPromptBuilder should default to include_reasoning=True."""
        builder = CodeActPromptBuilder()
        assert builder.include_reasoning is True

    def test_include_reasoning_true_has_thought_in_history(self):
        """With include_reasoning=True, history should include thought tags."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension", include_reasoning=True)

        obs = create_test_observation(1)
        msgs = builder.build_messages(obs)

        # Find the assistant message (action)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "<thought>" in assistant_msg["content"], "Should include <thought> tag"
        assert "</thought>" in assistant_msg["content"], "Should include </thought> tag"

    def test_include_reasoning_false_no_thought_in_history(self):
        """With include_reasoning=False, history should NOT include thought tags."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension", include_reasoning=False)

        obs = create_test_observation(1)
        msgs = builder.build_messages(obs)

        # Find the assistant message (action)
        assistant_msg = next(m for m in msgs if m["role"] == "assistant")
        assert "<thought>" not in assistant_msg["content"], "Should NOT include <thought> tag"
        assert "</thought>" not in assistant_msg["content"], "Should NOT include </thought> tag"
        # But should still have python tag
        assert "<python>" in assistant_msg["content"], "Should include <python> tag"

    def test_sequence_extension_with_no_reasoning_is_prefix(self):
        """Prompts with include_reasoning=False should still maintain prefix property."""
        builder = SimplePromptBuilder(prompt_mode="sequence_extension", include_reasoning=False)

        obs_0 = create_test_observation(0)
        obs_1 = create_test_observation(1)
        obs_2 = create_test_observation(2)

        text_0 = messages_to_text(builder.build_messages(obs_0))
        text_1 = messages_to_text(builder.build_messages(obs_1))
        text_2 = messages_to_text(builder.build_messages(obs_2))

        assert_is_char_prefix(text_0, text_1, "Step 0 should be prefix of Step 1")
        assert_is_char_prefix(text_1, text_2, "Step 1 should be prefix of Step 2")


class TestOriginalText:
    """The task, history and action-space text the harness renders, as the original ``CodeActAgent`` rendered it."""

    @pytest.mark.parametrize("max_steps", [None, 25])
    def test_task_text_is_the_original_task_str(self, max_steps):
        builder = CodeActPromptBuilder()
        goal = "Craft the following items: 3x a1_i5"

        assert builder.describe_task(goal, max_steps) == str(Task(goal=goal, max_steps=max_steps))

    def test_initial_messages_render_the_original_templates(self):
        builder = CodeActPromptBuilder(include_reasoning=False)
        state = CodeActState(task="Your Goal: g", action_space="ACTIONS")

        system, user = builder.build_messages(state)

        assert system["content"].startswith("# Instructions\n\nYou are a helpful agent")
        assert "<thought>" not in system["content"]
        assert user["content"] == (
            "# Task\n\nYour Goal: g\n\n# Action Space\n\nACTIONS\n\n"
            "Now, please provide the next python cell that you would like to execute. Output only one code block "
            "at a time formatted as follows <python>Code Block</python>."
        )

    def test_history_turns_strip_code_but_keep_thought_text(self):
        builder = CodeActPromptBuilder()
        state = CodeActState(task="t", history=[CodeActStep(code="x = 1", thought="\nwhy\n", output="", error="")])

        assistant, observation = builder.build_messages(state)[2:]

        assert assistant["content"] == "<thought>\nwhy\n</thought>\n<python>x = 1</python>"
        # The original template renders one newline before the request (trim_blocks eats the blank line).
        assert observation["content"] == "[Cell 0 Output]\n(No output)\nProvide your next action."

    def test_default_action_space_lists_signatures_and_descriptions(self):
        functions = [
            CellFunction("craft", [Parameter("ingredients", "dict"), Parameter("target", "list")], "Craft.", lambda: 0),
            CellFunction("launch_subagent", [Parameter("goal", "str")], "Delegate.", lambda: 0, is_async=True),
        ]

        text = CodeActPromptBuilder().describe_action_space(functions)

        assert text.startswith("Available Actions (python functions):\n\n1. def craft(ingredients: dict, target: list)")
        assert "2. async def launch_subagent(goal: str) -> str\n   Delegate." in text
        assert "**Do not forget to await**" in text
        assert CodeActPromptBuilder().describe_action_space([]) == ""

    def test_subagent_goal_is_the_targets_and_the_parents_context(self):
        builder = CodeActPromptBuilder()

        assert builder.subagent_goal("Summarize the file", "it is a csv") == (
            "Summarize the file\n\nContext provided from parent agent: it is a csv"
        )
        assert builder.subagent_goal({"stick": 2}) == "{'stick': 2}"  # TextCraft words its targets itself
        assert builder.subagent_task_misc({"stick": 2}, Task(misc={"a": 1})) is None  # the parent's misc


class TestReplies:
    """The original reply rule: stop at ``</python>``, put the tag back, then the original regexes."""

    @pytest.mark.parametrize(
        ("reply", "parsed"),
        [
            ("<thought>Bisect.</thought>\n<python>guess(50)</python>", ("guess(50)", "Bisect.")),
            ("<thought>Bisect.</thought>\n<python>guess(50)", ("guess(50)", "Bisect.")),  # cut at the stop
            ("<python>print(1)\n", ("print(1)\n", "")),
            ("<python>a</python> and some trailing text", ("a", "")),
            ("<thought>t</thought> <python>x = 1</python>", ("x = 1", "t")),  # not the original layout: found apart
            ("I will guess next time.", ("", "")),
            ("<thought>Only a thought.</thought>", ("", "Only a thought.")),
            ("", ("", "")),
        ],
    )
    def test_parse_reply(self, reply, parsed):
        assert CodeActPromptBuilder().parse_reply(reply) == parsed

    def test_every_call_stops_at_the_end_of_the_cell(self):
        assert CodeActPromptBuilder.stop == ("</python>",)

    def test_a_step_annotates_its_call_with_the_cell(self):
        step = CodeActStep(code="guess(5)", thought="t", output="Too low", error="")

        assert step.annotation() == {"thought": "t", "code": "guess(5)", "output": "Too low", "error": ""}
        assert list(CodeActStep().annotation()) == ["thought", "code", "output", "error"]
