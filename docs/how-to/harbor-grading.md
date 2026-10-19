# How Harbor tasks are graded

What platoon does between a harness's last step in a Harbor task's sandbox and the reward that trains: the steps
of a grade, what keeps the agents from forging their reward, and which reported reward trains.
[Sandboxes and Harbor tasks](sandboxes-and-harbor.md) shows how to train and evaluate on Harbor datasets; this page
is for writing tasks, reviewing a dataset, and reading the outcomes a run records.

## The steps of a grade

### In the agents' sandbox

Each rollout opens a sandbox from the task's image, creates `/logs/agent`, `/logs/verifier`, `/logs/artifacts` and
the workdir, waits for the task's healthcheck if it has one, runs the harness, then grades:

1. waits for every tool call in flight and holds the sandbox, so no agent's call runs until grading ends;
2. kills every process the agents left running in the sandbox, background and detached ones included;
3. checks the workdir: if the agents removed or moved it, left something else in its place or took its
   permissions, the attempt scores 0 (`workdir_missing`) and the tests do not run;
4. makes `/tests` and `/logs/verifier` afresh, and uploads `tests/` to `/tests` (agents never see it before this
   point);
5. runs `bash /tests/test.sh` in the workdir with the verifier's timeout and environment; if that command cannot
   start because the workdir went away after step 3 (exit status 126 on docker and hpc-sandbox), the attempt
   scores 0 (`workdir_missing`) and no reward file is read. A 126 the tests give themselves, with the workdir in
   place, is graded as usual. Where hpc-sandbox says the command never entered its working directory, the outcome's
   `verifier_cwd_error` metadata holds its line;
6. kills whatever the tests left running, then reads `reward.json` or `reward.txt` and the end of the verifier's
   logs from `/logs/verifier/`;
7. removes `/tests` again, and stops the sandbox when the rollout ends.

### In a separate verifier's sandbox

A task with a [separate verifier](sandboxes-and-harbor.md#a-separate-verifier) is graded the way Harbor grades it:

1. waits for the calls in flight and holds the agent's sandbox, kills every process the agents left running there,
   and checks the workdir as above (an unusable one scores 0, `workdir_missing`, and no verifier starts);
2. copies the `artifacts` and the contents of `/logs/artifacts` out, as directories and regular files only: a
   symlink, a device or a FIFO is left out, also below an artifact directory, and so is an artifact larger than
   `max_artifact_bytes`; the outcome's `artifacts_skipped` metadata names what was left out. A path that does not
   exist is skipped, and the tests decide what that means; one that exists but cannot be copied makes the outcome
   an `error` (`artifact_collect`);
3. stops the agent's sandbox right away when nothing else holds it (no shared fork, no call in flight). The session
   then takes no more tool calls or forks, and grading it again reuses the copied artifacts;
4. starts the verifier's sandbox (no healthcheck runs there, and only read-only `mounts` reach it), puts each
   artifact back at its path and `/logs/artifacts` back in place, and uploads `tests/` unless the image holds its
   own;
5. runs the tests there, kills whatever they left running, and reads `/logs/verifier/` as in the agent's sandbox,
   then stops the verifier's sandbox on every path, also when the rollout is cancelled. A verifier sandbox that
   cannot start makes the outcome an `error` (`verifier_start`).

Nothing an agent writes reaches the verifier except the artifacts' regular files, so a forged
`/logs/verifier/reward.txt` in the agent's sandbox counts for nothing. The outcome's metadata records
`verifier_mode` and the artifacts handed over. A verifier that times out makes the outcome `invalid`; it is not
trained on.

## How grading keeps the agent out

Grading takes the sandbox the tests run in away from the agents: their own sandbox for a task graded there
(Harbor's `shared` verifier mode), and the verifier's sandbox, where the tests may run the agent's code, for a task
with a separate verifier. What an agent might try instead of solving the task, and what grading does about it:

| What the agent tries | What grading does |
|---|---|
| Writes `reward.txt` or `reward.json` before grading, or leaves a symlink or a directory it took the permissions from (`chmod 000`) in `/logs/verifier` or `/tests` | Gives such a directory its permissions back, removes `/logs/verifier` and `/tests`, and makes them afresh before the tests run. A separate verifier never sees the agent's `/logs/verifier` |
| Leaves a process running that rewrites the reward files or the artifacts while the tests run, while the artifacts are copied, or after | Kills it before the tests run or the artifacts are copied, and kills whatever is left in the sandbox the tests ran in again before reading their output |
| Turns a reward file into a symlink to a file that reads `1`, or mounts such a file over it (from code the tests run) | Reads each reward file in the sandbox, only if it is a regular file reached through no symlink, with nothing mounted on or below `/logs/verifier`, and at most 1 MiB; anything else scores 0 (`reward_file_not_regular`, `reward_file_too_large`), a trainable outcome whose `reward_files` metadata names the file |
| Hands a separate verifier a symlink as an artifact, or inside one (to the tests' expected answer in the verifier's sandbox, or to a host file) | Copies artifacts as directories and regular files only; a symlink or a special file is left out (`artifacts_skipped`), so the verifier sees it missing |
| Leaves an artifact too large to copy, or one without permissions (`chmod 000`) | Leaves out an artifact over `max_artifact_bytes`, measured in the agents' sandbox before anything is copied, and gives one without permissions its owner's back before copying it: neither makes grading fail |
| Removes or moves its workdir, leaves a file or a symlink to nothing in its place, or takes its permissions (`chmod 000`) | Scores 0 without running the tests or starting a verifier (`workdir_missing`): a trainable outcome whose `workdir_state` metadata says what grading found. The tests run in the workdir (and the artifacts are copied from it), where no command can start now. Only grading treats it so: the agents' own commands fail from then on with exit status 126 (docker and hpc-sandbox, whose stderr says `hpc-sandbox: cannot enter the working directory /app: ...`) |

Two consequences for task authors. A task whose tests need a process the agent started, such as a server, cannot
pass: the tests must start what they check. And tests that run the agent's code run it with the verifier's
rights: a reward file that code leaves behind as anything but a regular file counts against the agent, but a
regular file it writes, or that a process it started writes before the tests end, counts as the tests' own, as
in Harbor. Agents that share a sandbox share its grading too: a grade, and the collection of a separate verifier's
artifacts, stops what every agent left running there, a harness still running in the sandbox included. So
`platoon train`, `eval` and `rollout` refuse `agents.child_grading: env` on `shared` forks of a Harbor sandbox,
which would grade a subagent in the sandbox its parent still works in
([How subagents are graded](../tutorials/subagents.md#how-subagents-are-graded)).

Grading cannot keep out a change to the programs in the agents' own sandbox. When a task is graded there, grading's
own steps run the sandbox's shell and tools (`sh`, `cat`), and the tests run its `bash` and whatever they call, such
as `python3` and the packages it imports. An agent that replaces one of them can forge its reward, as in Harbor: a
sandbox's root filesystem is writable, by root in a Docker container and by the user the agent runs as in Singularity
and enroot. So can an agent that leaves its answer as a symlink to a file in `/tests`, where the tests compare the
answer with that file. A task with a [separate verifier](sandboxes-and-harbor.md#a-separate-verifier) keeps both
out: its tests, and grading's reading of their output, run in the verifier's own image, and only the artifacts'
directories and regular files reach it. Where a policy may look for such shortcuts, train on tasks with a separate
verifier.

How each provider finds the processes to kill, and what it misses:

| Provider | Kills | Misses |
|---|---|---|
| `process` | each command's process group, and every process of yours whose environment holds the sandbox's `PLATOON_SANDBOX_ROOT` | a process that left its group and dropped that variable: the provider isolates nothing |
| `docker`, `singularity`, `enroot` | every process in the sandbox's PID namespace but the ones that keep the container alive, which it lists when the sandbox starts (Docker's `sleep`; Singularity's `sinit`, keep-alive script and `sleep`; enroot's keep-alive loop and `sleep`); it freezes them all at once first, so none forks past it | nothing it can see. Where it cannot tell what to kill, the grade is an `error` (`verifier_isolation`) instead: in a sandbox whose `/proc` does not list its processes (root in an enroot container can mount over it), and in one without a PID namespace of its own (Singularity where the site sets `allow pid ns = no`), where the kill would reach your processes outside it |

The forwarder of a [gateway route](sandboxes-and-harbor.md#3-pick-where-the-harness-runs) is killed with the
rest; the runner starts it again before the next harness launches. A provider of your own gives its sandboxes a
`kill_processes()` method; without one, grading logs a warning and runs the tests with the agents' processes still
running.

## Which reward trains

An entry's [reward contract](sandboxes-and-harbor.md#rewards-the-reward-contract-and-log-classifiers) names the
reward that trains, or lists the names a dataset's tasks report it under (`reward: [reward, correctness]`):

| A reading that... | Is |
|---|---|
| reports exactly one listed name, from one reward file | scored under that name, which the outcome records as its `primary` |
| reports none of them | an error (`verifier_no_primary`), never 0 |
| reports two or more of them | `invalid` (`verifier_ambiguous_primary`) |
| comes with both `reward.json` and `reward.txt` | `invalid` (`verifier_ambiguous_reward_files`) |

So the list's order never matters, and code of the policy's that the tests run cannot choose which name trains by
writing a second key or a second file. `range` bounds the name that trained, and `components`, when set, must
include every listed name. With one name, a reading is graded as before, `reward.json` first; a list of one name
(`primary: [correctness]`) follows the list's rules, so a second reward file makes the reading `invalid`.

**One name per group.** A group's rollouts are centered on each other's rewards, so they must have trained the same
name: a group whose rollouts trained different listed names does not train, and counts as `reward_key_mismatch`.
Which name a task's tests report is the task's, and this keeps a policy whose code they run from choosing it rollout
by rollout. Each rollout's record names the name it trained (`misc.outcome.reward_key` of its root trajectory, and
`reward_key` in `platoon eval`'s `metadata.json`), and each step counts its rollouts by it
(`train/batch/reward_key/<name>`).

The zeros platoon gives itself, for a reward file that is not a regular file and for an agent that made its
workdir unusable ([How grading keeps the agent out](#how-grading-keeps-the-agent-out)), score 0 under the
contract's first name. No grader reported that name: such an outcome's metadata says `platoon_zero: true`, it
trained no reward key (`reward_key: null`), and it takes no part in a group's check.

## Next steps

- [Sandboxes and Harbor tasks](sandboxes-and-harbor.md): sandbox providers, datasets, rewards, images, harnesses in
  task sandboxes, and what platoon refuses to run.
- [Troubleshooting](troubleshooting.md): missing or rejected reward files, and the other outcomes a grade records.
- [Write a sandbox provider](../extending/sandbox-providers.md): sandboxes on infrastructure of your own, with the
  `kill_processes()` that grading relies on.
