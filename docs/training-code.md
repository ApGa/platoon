# The training code

This page is a map of `platoon/train`, the code behind `platoon train`, for contributors who read or change the
trainer. It follows one start of a run from the command line to its first step, then one step from the draw of a task
to the new policy and the checkpoint, and where every step ends, with a link to the code of each part.

Read first: [One training step](architecture.md#one-training-step) and
[Rollout workers](architecture.md#rollout-workers) in the architecture, and the words the training code uses
throughout, defined in the docstring of [`platoon.train`](../platoon/train/__init__.py) (a start, a group, a draw, to
settle, the horizon, an entry, a label). The [glossary](#glossary) at the end defines the others.

## Where the code runs

- **The trainer process** runs `platoon train`: the dispatcher that starts the rollout groups, the workflow that runs
  each group, the training steps, the evaluation loop and the saves. With `train.workflow_config.rollout_isolation:
  inprocess` it runs the rollouts too.
- **Rollout workers** run one rollout each, with `rollout_isolation: process`. A fork server per workflow imports the
  workers' code ahead of them, and forks each worker. A worker reads its request from a file, runs the rollout, writes
  its response and exits ([`platoon/train/workers/`](../platoon/train/workers/)).
- **The training service** is a Tinker-API service: hosted Tinker, or a server that implements its API. The trainer
  holds a training client of it per policy (forward-backward passes, optimizer steps, saves), and every rollout samples
  through a sampling client of the newest saved weights.

```text
trainer process (platoon train)                          training service (Tinker API)
+-------------------------------------------+            +------------------------------+
| TrainerRun -> PlatoonTinkerRLTrainer      |            | a training client per policy |
|   Dispatcher -> GroupRolloutWorkflow      |            |   forward-backward passes,   |
|     a group's members, in process   ------+-- sample ->|   optimizer steps, saves     |
|   training loop -> TrainStep -> Learner --+-- train -->| a sampling client of the     |
|   evaluation loop                         |            |   newest saved weights       |
+-------------------+-----------------------+            +------------------------------+
                    | worker request and response files                ^
                    v                                                  |
rollout workers (rollout_isolation: process)                           |
  one per rollout, forked by a fork server per workflow  ---- sample --+
```

That is an rl run. An sft run (`trainer: sft`) has the trainer process and the training service only: it rolls nothing
out, so it has no dispatcher, workflow, rollout workers or evaluation loop.

## The map

Each package is one stage of a run; the modules at the top set a run up.

| Path | What it holds |
|---|---|
| [`launch.py`](../platoon/train/launch.py), [`supervise.py`](../platoon/train/supervise.py) | `platoon train`'s start: load and check the config, list the tasks, build the trainer and its workflows (a trainer that rolls nothing out, sft, lists no task and builds none); run it in its own process, or supervised and restarted |
| [`trainer/`](../platoon/train/trainer/) | the trainer API and the lifecycle every trainer goes through (`api.py`, `run.py`), each trained policy's own clock when they step independently (`clocks.py`), the step hooks, work at the end of each step of any trainer (`hooks.py`; `hook_config.py` attaches those a config names to the rl trainer), the `trainer:` key, core's trainers and their options (`registry.py`), the rl trainer (`rl.py`), its training loop (`loops.py`) and the state its loops share (`state.py`), the sft trainer (`sft.py`), signals and the watchdog (`signals.py`), waiting with heartbeats (`waiting.py`) |
| [`policy/`](../platoon/train/policy/) | one model that training samples and steps: its spec, its state, its training service and saves, and its learner; and the run's policies, with which policy each agent samples |
| [`objectives/`](../platoon/train/objectives/) | what a policy's rows train under: the loss and its training metrics, the rl trainer's (`policy_gradient.py`) and the sft trainer's (`cross_entropy.py`) |
| [`step/`](../platoon/train/step/) | one training step of the rl trainer (`step.py`) and its stages; each trained policy's lane through it (`lanes.py`), and each policy's loop when they step independently (`independent.py`) |
| [`dispatch/`](../platoon/train/dispatch/) | the run's draws (`source.py`): the dispatcher that starts the training groups, the rollout permits, a group's run and how it settles; and, when the policies step independently, each policy's horizon and the ledger of its shares of the groups (`horizons.py`, `shares.py`) |
| [`workflow/`](../platoon/train/workflow/) | one group of rollouts: the workflow contract (`protocol.py`) and `GroupRolloutWorkflow` with its parts |
| [`credit/`](../platoon/train/credit/) | a group's credit: its baselines, error-token filter and verdict (`group.py`), and the credit estimators an entry may name instead (`api.py`, `registry.py`, `runtime.py`, and core's) |
| [`workers/`](../platoon/train/workers/) | process-per-rollout: the worker's entrypoint, its supervisor and the protocol between them |
| [`checkpoints/`](../platoon/train/checkpoints/) | what a full checkpoint saves, and how a start resumes from it: a barrier of every trained policy's saves (`barrier.py`), or, when they step independently, a commit of one policy's (`commits.py`) |
| [`evaluation/`](../platoon/train/evaluation/) | evaluation during training |
| [`supervised/`](../platoon/train/supervised/) | the sft trainer's datasets, in LLaMA-Factory's conventions: its options, a dataset directory's `dataset_info.json`, the row formats, the loading, a conversation rendered into the datum it trains on, and the cursor that walks the conversations epoch by epoch |
| `config_defs.py`, `components.py`, `entry_settings.py`, `auto.py`, `labels.py`, `exclusion.py`, `identity.py` | the config's schema, each environment's components and settings, the label column, excluded tasks, one run per run directory |
| `dataloader.py`, `dataset_order.py`, `startup.py`, `reaping.py`, `leases.py` | the training tasks in batches and their order, what a run readies before its first rollout, what dead processes left, rollout leases |
| `sampler_api.py`, `samplers.py`, `sampler_runtime.py`, `sampling.py`, `admission_api.py`, `admissions.py`, `task_stats.py`, `task_records.py` | task selection (RFC 0019): which task each group runs, which groups count, and the per-task statistics |
| [`tinker/`](../platoon/train/tinker/) | the policy client ([`proxy.py`](../platoon/train/tinker/proxy.py)), its renderers and the rollout timing; and the shims of the module paths that platoon 0.5.0 moved, which platoon 0.6.0 removes |

The [dependency rule](../tests/train/test_layers.py) keeps the layers apart: generic code (`policy/`, `objectives/`,
`dispatch/`, `checkpoints/`, `evaluation/`, `workers/`, `supervised/`, the trainer API, lifecycle and registry with the
step hooks, each policy's own clock, the signals and the waits, `trainer/api.py`, `run.py`, `registry.py`, `hooks.py`,
`clocks.py`, `signals.py` and `waiting.py`, and the flat modules but `launch.py`) imports no trainer and no plugin; the
plugins (`workflow/`, `credit/`) import no trainer; the trainers, rl's (`trainer/rl.py`, `loops.py`, `state.py`,
`step/`) and sft's (`trainer/sft.py`), may import anything.

## Policies

A policy is one model that training samples and steps.

- [`PolicySpec`](../platoon/train/policy/spec.py) is a policy as the config declares it, read once as the trainer is
  built ([`policy_specs`](../platoon/train/policy/spec.py)): the model and renderer, the service's URL, the LoRA rank,
  the optimizer's settings, the loss and its config, the checkpoint it starts from. `main` is `train:`'s, and
  `policies:` declares the run's others: frozen ones that its agents sample or a credit estimator scores with, and
  others it trains beside `main`, in lockstep or each on its own clock (`train.stepping`). Which policy each agent
  samples is `agents:`'s ([`AgentBindings`](../platoon/train/policy/spec.py)).
- [`Policy`](../platoon/train/policy/policy.py) holds its state as training moves it on: its model info (the model
  gateway that samples it, with its version), its Tinker clients, its optimizer's parameters and the optimizer steps
  it took. [`Policy.open`](../platoon/train/policy/policy.py) opens its training client at a start point, and
  [`Policy.save`](../platoon/train/policy/policy.py) and
  [`Policy.save_for_sampler`](../platoon/train/policy/policy.py) save it whole or for its sampler
  ([`policy/checkpoints.py`](../platoon/train/policy/checkpoints.py)).
- [`PolicySet`](../platoon/train/policy/set.py) is the run's policies, with their leader, `main`. Each policy's
  version is the count its model gateway keeps ([`PolicySet.versions`](../platoon/train/policy/set.py)), which each
  publication of a new sampling client counts up ([`Policy.publish`](../platoon/train/policy/policy.py)).
- [`open_training_client`](../platoon/train/policy/service.py) asks the service for a training client until it accepts
  one, for up to 20 minutes: a service may refuse a new start while it holds a dead start's session.
- [`_publish_worker_binding`](../platoon/train/policy/binding.py) hands each new sampling client to a workflow's rollout
  workers, which follow it ([`FollowingSamplingClient`](../platoon/train/workers/binding.py)).

## The trainer lifecycle

`platoon train` runs [`arun_tinker_training`](../platoon/train/launch.py): it loads and checks the config, lists the
tasks, builds the components of each environment (the Auto factories of [`auto.py`](../platoon/train/auto.py)), builds
the rl trainer ([`PlatoonTinkerRLTrainer`](../platoon/train/trainer/rl.py)) and its workflows on the trainer's policy,
and calls [`PlatoonTinkerRLTrainer.train`](../platoon/train/trainer/rl.py). The config's `trainer:` key names the
trainer, one of core's ([`CORE_TRAINERS`](../platoon/train/trainer/registry.py)): `rl`, the default, or `sft`
([`SupervisedTrainer`](../platoon/train/trainer/sft.py)), which rolls nothing out, so that `platoon train` imports no
environment and lists no task for it, and builds it from its setup and runs it
([`_train_without_rollouts`](../platoon/train/launch.py)).

Every start of every trainer goes through one lifecycle, [`TrainerRun.execute`](../platoon/train/trainer/run.py), which
calls the trainer's hooks ([`TrainerBase`](../platoon/train/trainer/api.py)) in this order:

0. the start's workflows, when the trainer rolls out and was given none;
1. `check`: the trainer's checks, where it may register step hooks (`run.hooks`) and participants of the run's
   checkpoints (`run.checkpoints`); the rl trainer attaches the hooks its options name
   ([`attach_hooks`](../platoon/train/trainer/hook_config.py));
2. the signal handlers ([`_setup_signal_handlers`](../platoon/train/trainer/signals.py)): a first signal asks the
   steps to stop at the next step boundary, a second forces the exit;
3. each policy's service, and where the start begins ([`StartPoint`](../platoon/train/checkpoints/resume.py): the
   checkpoint it resumes from, and each policy's counters there);
4. `_open_policies`: each policy's training client at the start point;
5. `prepare`: the trainer readies its steps and says where they end; a trainer that rolls out opens the start's draws
   here ([`TrainerRun.rollouts`](../platoon/train/trainer/run.py)). Then every key of the checkpoint the start resumes
   from must be claimed, by the run's counters or by a participant
   ([`TrainerRun._check_claimed`](../platoon/train/trainer/run.py));
6. the launch save, a recovery checkpoint, then `started`, and each policy's first sampling client, which the step
   hooks hear of;
7. `ready`: the run's clock starts ([`StepClock`](../platoon/train/trainer/run.py));
8. `run_steps`: the steps, each of which ends the same way ([the end of a step](#the-end-of-a-step), below);
9. the save at exit: `<step>_interrupted` after an interrupt, `final` at the run's end; an interrupt inside a step
   that no checkpoint may save saves nothing ([`TrainerRun._unsaved_step`](../platoon/train/trainer/run.py));
10. in `finally`, also after a failure: `close`, then each step hook's and participant's `aclose`, in reverse order of
    joining ([`TrainerRun._close`](../platoon/train/trainer/run.py)).

The rl trainer's lifecycle hooks are its methods of those names: `prepare` begins the rollouts' startup
([`RolloutStartup`](../platoon/train/startup.py): the harness toolchains staged, the workers' fork servers started) and
opens the draws over its tasks; `started` starts the pruning of each trained policy's old recovery checkpoints
([`RecoveryCheckpointPruner`](../platoon/train/checkpoints/pruning.py)); `ready` builds what the loops share
([`TrainLoopSharedState`](../platoon/train/trainer/state.py), over the run's policies and clock, and each trained
policy's own clock when they step independently); and `run_steps` runs
the training loop ([`_train_loop`](../platoon/train/trainer/loops.py)) and, beside it, the evaluation loop
([`_eval_loop`](../platoon/train/evaluation/loop.py)), until the training loop stops both
([`shutdown_loops`](../platoon/train/trainer/loops.py)).

The sft trainer's ([`SupervisedTrainer`](../platoon/train/trainer/sft.py)) train on conversations instead of rollouts:
`check` loads the datasets its options name ([`load_conversations`](../platoon/train/supervised/loading.py)), renders
those that show what they ask of the policy's renderer ([`check_renders`](../platoon/train/supervised/rendering.py))
and registers the cursor that walks them with the run's checkpoints
([`DatasetCursor`](../platoon/train/supervised/cursor.py)); `prepare` counts the run's end on from the cursor;
`started` starts the pruning of old recovery checkpoints; `ready` takes the policy's learner from the run
(`run.learner`) under cross-entropy; and `run_steps` runs its steps
([`SupervisedTrainer.step`](../platoon/train/trainer/sft.py)): the next `batch_size` conversations, each rendered into
the datum it trains on ([`supervised_datum`](../platoon/train/supervised/rendering.py)), one optimizer step over them,
then the step's end ([below](#the-end-of-a-step)).

## The rollout source

A trainer that trains on rollouts of the run's tasks draws them through the start's rollout source,
[`RolloutSource`](../platoon/train/dispatch/source.py). It holds one start's draws:

- the run's per-task statistics ([`TaskRecorder`](../platoon/train/task_records.py)): every draw, every group that
  settled and its fate;
- the order of the training tasks ([`PlatoonTinkerDataloader`](../platoon/train/dataloader.py)), and the task samplers
  that draw it ([`SamplerSet`](../platoon/train/sampler_runtime.py));
- the group admission rules ([`AdmissionSet`](../platoon/train/admissions.py)), if any entry has one;
- the dispatcher ([`Dispatcher`](../platoon/train/dispatch/dispatcher.py)), which starts each draw's group when the
  horizon, the rollout permits ([`PermitPool`](../platoon/train/dispatch/permits.py)) and the label's limit allow;
- once started ([`RolloutSource.start`](../platoon/train/dispatch/source.py)), the queue its groups settle into.

[`RolloutSource.open`](../platoon/train/dispatch/source.py) opens them before the launch save, and the run's
checkpoints take the source in as a participant, so every full checkpoint saves where its draws stand
([`RolloutSource.checkpoint_state`](../platoon/train/dispatch/source.py)), and a start that resumes takes that up
([`RolloutSource.restore`](../platoon/train/dispatch/source.py)). Each started group runs in
[`_run_training_group`](../platoon/train/dispatch/runner.py): its workflow, awaited under the watchdog's heartbeat
([`run_group_guarded`](../platoon/train/dispatch/runner.py)), its record in the statistics, and its settling as a
[`SettledGroup`](../platoon/train/dispatch/settled.py), or, under a group admission rule, its decision first
([`_settle_under_rule`](../platoon/train/dispatch/runner.py)). A step that consumed groups commits them
([`RolloutSource.committed`](../platoon/train/dispatch/source.py)), and the next step moves the horizon on
([`RolloutSource.step_started`](../platoon/train/dispatch/source.py)).

## A group of rollouts

The workflow contract ([`RolloutWorkflow`](../platoon/train/workflow/protocol.py)) is all the trainer needs of a
workflow: one row in, the group's output out. The trainer reads every output through the contract's readers:
[`rows_by_policy`](../platoon/train/workflow/protocol.py) (its datums, by the policy each trains) and
[`report_of`](../platoon/train/workflow/protocol.py) (its label, statistics and workload).

The run's workflow is [`GroupRolloutWorkflow`](../platoon/train/workflow/group.py):

1. [`GroupRolloutWorkflow.arun_episode`](../platoon/train/workflow/group.py) runs the group with its entry's settings
   ([`workflow/context.py`](../platoon/train/workflow/context.py)).
2. [`GroupRolloutWorkflow._run_group`](../platoon/train/workflow/group.py) runs the members
   ([`_run_group_rollouts`](../platoon/train/workflow/members.py): the straggler policy's quorum, grace and cut), under
   the per-datum stale filter stamps each member's datums with their own versions
   ([`GroupRolloutWorkflow._stamp_own_versions`](../platoon/train/workflow/group.py), below), then computes the
   group's credit.
3. Each member runs in a block of its draw's rollout lease ([`_run_group_member`](../platoon/train/workflow/members.py);
   [`rollout_lease`](../platoon/train/leases.py)), after the startup gate of process isolation
   ([`_enter_startup_gate`](../platoon/train/workflow/members.py)) and a start slot of `train.max_starting_rollouts`
   ([`StartLimit`](../platoon/train/workflow/start_limit.py)).
4. One member's rollout is [`arun_episode_single`](../platoon/train/workflow/members.py), in process
   ([`run_in_process`](../platoon/train/workflow/members.py)) or in a rollout worker
   ([`run_in_worker`](../platoon/train/workflow/members.py),
   [`RolloutWorkerSupervisor`](../platoon/train/workers/supervisor.py)).
   Either way it runs [`run_rollout_request`](../platoon/train/workers/run.py) on a worker request
   ([`_build_worker_request`](../platoon/train/workflow/requests.py)), which runs the environment's rollout (by
   default [`harness_rollout`](../platoon/harness/rollout.py)) and converts its trajectories into datums
   ([`get_train_data_for_trajectory_collection`](../platoon/utils/tinker_data_processing.py)), each stamped with the
   rollout's policy version: the oldest any of its samples used
   ([`rollout_version`](../platoon/train/workers/run.py)); each datum's own version, the oldest its own model calls
   sampled, goes beside them
   ([`TrajectoryCollectionResult.own_versions`](../platoon/utils/tinker_data_processing.py)).
5. The credit ([`credit/group.py`](../platoon/train/credit/group.py)): the completed roots set the baseline
   ([`group_baselines`](../platoon/train/credit/group.py)), every datum is centered on it, the error tokens are filtered
   ([`filter_error_tokens`](../platoon/train/credit/group.py)), and the group's verdict says what it trains
   ([`group_verdict`](../platoon/train/credit/group.py)). An entry may name a credit estimator instead
   ([`CreditEstimator`](../platoon/train/credit/api.py)), which assigns the credit over the group its members' rows
   make ([`_credit_group`](../platoon/train/workflow/crediting.py)); in a run that trains several policies every entry
   has one (`keyed`, unless it names another).
6. The output is a [`GroupOutput`](../platoon/train/workflow/output.py): its datums, with the group's workload,
   statistics, label and what each member returned ([`MemberOutcome`](../platoon/train/workflow/output.py)); or, when
   its rows carry tags (an estimator that reads agents, or a run that trains several policies,
   [`tags_rows`](../platoon/train/workflow/crediting.py)), a
   [`PolicyGroupOutput`](../platoon/train/workflow/output.py), its rows tagged by the policy, member and agent that
   produced them.

## The learner and the objective

A step updates a policy through its [`Learner`](../platoon/train/policy/learner.py), once per minibatch: a
[`PolicyUpdate`](../platoon/train/policy/learner.py) sends each microbatch's forward-backward pass as it is ready
([`PolicyUpdate.submit`](../platoon/train/policy/learner.py)), then steps the optimizer once over them all and records
the update's metrics ([`PolicyUpdate.apply`](../platoon/train/policy/learner.py)). Each result is awaited under its
`train.*` deadline. A learner's first submission of a step opens the step (the clock's `step_open`), a policy has one
update open at a time, and a learner that fails closed refuses an optimizer step that the service skipped or whose
metrics are not finite ([`UpdateRejected`](../platoon/train/policy/learner.py)), as a trainer's needs ask
([`TrainerNeeds`](../platoon/train/trainer/api.py): the sft trainer's do, the rl trainer's do not).

An rl step's rows train under the policy's objective,
[`PolicyGradient`](../platoon/train/objectives/policy_gradient.py): the loss its spec names (`cispo`, `ppo` or
`importance_sampling`) with its config, the loss inputs only the trainer reads, which the datums sent leave out
(`side_channels`), and each microbatch's training metrics
([`compute_training_metrics`](../platoon/train/objectives/policy_gradient.py)). An sft step's train under
[`CrossEntropy`](../platoon/train/objectives/cross_entropy.py): each datum's weights mark the tokens that train, scaled
so that the loss is the mean over the step's trained tokens
([`mean_per_token`](../platoon/train/objectives/cross_entropy.py)).

## One step, from the draw to the checkpoint

The training loop ([`_train_loop`](../platoon/train/trainer/loops.py)) starts the draws, publishes the first sampler of
each policy the agents sample to the rollout workers, builds what its steps share
([`StepContext`](../platoon/train/step/step.py)) and runs one [`TrainStep`](../platoon/train/step/step.py) per step.

Beside the steps, the dispatcher draws: it draws the next position of the order through the task samplers and starts
its group when the horizon (at most `(train.max_staleness + 1) × batch_size` groups started and not yet consumed; with
several trained policies, the smallest bound of those the agents sample,
[`horizon_bound`](../platoon/train/policy/spec.py)), the rollout permits and the label's limit allow
([`Dispatcher.run`](../platoon/train/dispatch/dispatcher.py)). The group runs, as above, and settles into the queue the
steps dequeue from.

[`TrainStep.run`](../platoon/train/step/step.py) reads top to bottom, in seven stages:

1. **Collect.** The step dequeues settled groups into microbatches
   ([`TrainStep.collect_microbatch`](../platoon/train/step/step.py)). Each group is tallied
   ([`StepTally`](../platoon/train/step/collect.py)), counted under a group admission rule
   ([`GroupCounting`](../platoon/train/step/counting.py)), dropped when its oldest datum is more versions old than its
   entry's bound ([`StaleFilter`](../platoon/train/step/staleness.py)), or under `train.stale_filter: datum` stripped
   of the datums that are ([`TrainStep._filter_stale`](../platoon/train/step/step.py),
   [below](#versions-and-staleness)), and given its fate ([`TrainStep.take_group`](../platoon/train/step/step.py)).
   In a run of several trained policies, each policy's rows of a group go their own lane through the stages
   ([`PolicyLane`](../platoon/train/step/lanes.py)), judged against that policy's version and bound.
2. **Prepare.** Each microbatch goes through the batch transforms
   ([`run_batch_transforms`](../platoon/train/step/batch_transforms.py), such as depth weighting) and the zero-advantage
   filter, and is sent without the objective's side channels, its advantages scaled by the action tokens it represents
   ([`TrainStep.prepare`](../platoon/train/step/step.py)).
3. **Update.** One optimizer update per minibatch, through each trained policy's learner
   ([`TrainStep.update`](../platoon/train/step/step.py)). A fresh run's first update, a step hook's
   too, waits for the evaluation of its starting policy, with `eval.at_start`: the learner's barrier
   ([`_starting_policy_barrier`](../platoon/train/step/step.py)), which
   [`PolicyUpdate.apply`](../platoon/train/policy/learner.py) awaits.
4. **Commit.** The step commits what became of each of its groups to the draws
   ([`TrainStep.commit`](../platoon/train/step/step.py)), before any checkpoint saves them.
5. **End.** The step ends as every trainer's does ([`finish_step`](../platoon/train/trainer/hooks.py), below): the
   run's step hooks drain and update the policies, then each policy that the step or a hook updated is saved once, a
   checkpoint of every trained policy when one is due, and its new sampling client goes to the policy and the rollout
   workers ([`TrainStep.publish`](../platoon/train/step/step.py),
   [`publish_policies`](../platoon/train/step/publish.py),
   [`_save_checkpoint_and_get_sampling_client`](../platoon/train/checkpoints/saving.py) for a run's one trained
   policy, [`save_after_step`](../platoon/train/checkpoints/saving.py) for several), and the hooks hear of it. A
   policy that took no update says why and keeps its sampler.
6. **Advance.** Last, the end moves the run's clock on ([`StepClock.advance`](../platoon/train/trainer/run.py)): the
   horizon moves one batch, the evaluation loop wakes and the watchdog is fed.
7. **Log.** The step's metrics are logged ([`TrainStep.log`](../platoon/train/step/step.py),
   [`step/metrics.py`](../platoon/train/step/metrics.py)).

When the trained policies step independently (`train.stepping: independent`), the training loop runs one loop per
policy instead ([`train_policies`](../platoon/train/step/independent.py)), each on its own clock
([`OwnClocks`](../platoon/train/trainer/clocks.py)). Each of its steps is the step above with the policy's one lane
([`PolicyStep`](../platoon/train/step/independent.py)), which takes the policy's shares of the settled groups from the
share ledger ([`Shares`](../platoon/train/dispatch/shares.py)) in place of the queue, and publishes that policy alone.
The dispatcher keeps a horizon per policy ([`Horizons`](../platoon/train/dispatch/horizons.py)).

## The end of a step

Every step of every trainer ends in one sequence, [`finish_step`](../platoon/train/trainer/hooks.py), after the step's
own updates and its commit. The rl trainer's step runs it ([`TrainStep.run`](../platoon/train/step/step.py)); a trainer
of its own runs it through [`TrainerRun.end_step`](../platoon/train/trainer/run.py), which counts the step's updates
from each policy's optimizer steps and publishes through
[`TrainerRun.publish`](../platoon/train/trainer/run.py): a full checkpoint when the cadence says one is due, else a
weights-only save, and the new sampler to the rollout workers. The sft trainer, whose policy nothing but a step hook
samples, ends its steps with `run.end_step(sampled=False)`: a step without hooks, or one that neither it nor a hook
updated, publishes nothing and saves only the full checkpoint the cadence asks for
([`TrainerRun._save_due`](../platoon/train/trainer/run.py)). A recovery checkpoint that a step's end saves goes to the
trainer's pruning ([`TrainerBase._recovery_saved`](../platoon/train/trainer/api.py)).

The step hooks are the run's ([`StepHooks`](../platoon/train/trainer/hooks.py), `run.hooks`), each a
[`StepHook`](../platoon/train/trainer/hooks.py) ([`StepHookBase`](../platoon/train/trainer/hooks.py) gives each
method a default). With hooks, the end:

1. tells each hook what the step did, in a [`StepEnd`](../platoon/train/trainer/hooks.py): the step, its own updates,
   the draws it committed with their fates, the run's policies and its waits
   ([`TrainerRun._step_end`](../platoon/train/trainer/run.py));
2. lets each hook drain its side work, then update the policies, through `end.train` or `end.learner`
   ([`StepEnd.learner`](../platoon/train/trainer/hooks.py): a learner on the policy's training client and Adam
   state, which sends and counts only during the hook's `update`);
3. publishes each policy that the step or a hook updated, once, as one new version, so that the hooks spend none of
   the staleness budget;
4. tells each hook what was published (`published`), also when nothing was;
5. advances the run's clock.

With no hook registered it publishes and advances, nothing else. Meanwhile the clock says whether the step sent
training work (`step_open`) and whether its end runs the hooks (`ending`), by which the save at exit knows a step that
no checkpoint may save (rule 10 of RFC 0021). The rl trainer attaches the hooks its config names
(`trainer_options.hooks`); [Step hooks](extending/step-hooks.md) is the guide to writing one.

## Beside the step

- **Evaluation**, the rl trainer's ([`EvaluationLoop`](../platoon/train/evaluation/loop.py)), wakes at each step's end
  and evaluates the step training reached when its evaluation is due
  ([`TrainLoopSharedState.evaluation_due`](../platoon/train/trainer/state.py)),
  its groups run by [`eval_worker`](../platoon/train/evaluation/loop.py). It holds a fresh run's first update until
  step 0 is evaluated ([`_wait_for_starting_policy_evaluation`](../platoon/train/evaluation/loop.py)), and the loops
  stop only once it has handled the last step ([`_wait_for_last_step_evaluation`](../platoon/train/evaluation/loop.py)).
  Each evaluation that completes is recorded in `evaluations.jsonl`
  ([`record_evaluation`](../platoon/train/evaluation/record.py)), so that a resume does not run it again.
- **Pruning** deletes the run's older recovery checkpoints beside training, keeping those a resume or a slow group may
  still need ([`RecoveryCheckpointPruner`](../platoon/train/checkpoints/pruning.py)); an sft run's keeps none for a
  sampler, since no rollout samples its policy.
- **Signals and the watchdog**: a first SIGINT or SIGTERM stops the steps at the next boundary, and the run saves where
  it is; the [`Watchdog`](../platoon/train/trainer/signals.py) ends a process that stops making progress, which every
  long wait feeds ([`_await_with_heartbeat`](../platoon/train/trainer/waiting.py)).
- **Throughput**: the rollout client's metrics ([`rollout_metrics.py`](../platoon/train/tinker/rollout_metrics.py)) say
  how many rollouts ran at once and where a step's time went.

## Saving and resuming

A full checkpoint is the run's ([`RunCheckpoints.save`](../platoon/train/checkpoints/participants.py)): a row of
`checkpoints.jsonl` whose `loop_state` holds the step a resume starts at, the W&B run, the leader policy's version and
optimizer steps, then each participant's entries (the draws': the statistics' snapshot, the dispatcher's state in a
file of its own with the groups that settled and no step consumed,
[`checkpoints/dispatch_state.py`](../platoon/train/checkpoints/dispatch_state.py),
[`checkpoints/kept_groups.py`](../platoon/train/checkpoints/kept_groups.py); the sft trainer's cursor's, `sft`: the
epoch and the position in it, and what decides its order, the conversations and the seed); and the leader policy's
save of its weights and optimizer state. In a run of several trained policies it is a barrier of every one's save,
which the leader's row, written last, commits, naming every other policy's row (a start resumes from the newest one
whose named rows hold, [`newest_barrier`](../platoon/train/checkpoints/barrier.py)); when they step independently,
every full save of any policy is a commit of its own, which names the row of every other policy that a resume from it
starts at ([`Commits`](../platoon/train/checkpoints/commits.py),
[`newest_commit`](../platoon/train/checkpoints/commits.py)).

After each step that updated, the cadence ([`checkpoint_kind_for_step`](../platoon/train/checkpoints/saving.py)) saves
a permanent checkpoint every `checkpoint.every` steps, a recovery one every `checkpoint.recovery_every`, and otherwise
the weights only, for the sampler; an sft step without a step hook saves only the cadence's checkpoints. A start
resumes from the newest row that holds a state ([`StartPoint.read`](../platoon/train/checkpoints/resume.py)); its draws
take up what the row saved ([`_restore_dispatch`](../platoon/train/checkpoints/resume.py)): the groups it kept train as
they are, and the draws it left outstanding are drawn again. The sft trainer's cursor takes up its epoch and position
where the row left them, and refuses a row that walked another order: other conversations, or another seed
([`DatasetCursor.resume`](../platoon/train/supervised/cursor.py)).

## Versions and staleness

- A rollout's datums carry one policy version, their `checkpoint_version` loss input, stamped as the rollout's
  trajectories become datums: the oldest version any sample of the rollout used, in a worker that followed the
  trainer's newer weights while it ran, else the one its request was bound to
  ([`rollout_version`](../platoon/train/workers/run.py)).
- Each publication counts the policy's version up by one ([`Policy.publish`](../platoon/train/policy/policy.py)).
- The horizon bounds how far rollouts run ahead of training: with `train.max_staleness: k`, at most `k + 1` batches of
  groups are started and not yet consumed.
- The stale filter drops a group whose oldest datum is more than its entry's bound behind the policy's version
  ([`StaleFilter.check`](../platoon/train/step/staleness.py), [`datum_version`](../platoon/train/step/staleness.py)).
  In a run of several trained policies, each policy's rows of a group are judged against its own version and bound
  ([`StaleFilter.for_policy`](../platoon/train/step/staleness.py)), and only its stale rows are dropped.
- Under `train.stale_filter: datum` (experimental) it judges each datum instead. Each model call records the version
  it sampled (`ChatResponse.version`; in a rollout worker, the version of the binding that its
  [`FollowingSamplingClient`](../platoon/train/workers/binding.py) handed out with the client), and the conversion gives
  each datum its own version, the oldest among its calls (`own_versions`). The trainer tells the training workflow
  ([`_set_stale_filter`](../platoon/train/trainer/rl.py)), and `GroupRolloutWorkflow` stamps each member's datums with
  their own versions before the credit
  ([`TrajectoryCollectionResult.stamp_own_versions`](../platoon/utils/tinker_data_processing.py)); a workflow that does
  not take the setting is judged by the versions its datums carry. The step judges the datums
  ([`StaleFilter._check_datums`](../platoon/train/step/staleness.py)): a group none of whose datums is within the bound
  is dropped whole; of the others it masks the datums beyond it
  ([`TrainStep._count_masked`](../platoon/train/step/step.py)) and trains the rest with the advantages their group's
  credit gave them ([`_kept_datums`](../platoon/train/step/datums.py), which moves a masked trajectory start to the
  next kept datum), counting what it dropped in `train/batch/stale_datums`.

## Invariants a change must keep

- Every draw is consumed exactly once: a full checkpoint saves the draws no committed step consumed, and a resume
  draws them again or trains the groups it kept.
- With `train.max_staleness: 0` a step trains its own draws, unless a label's limit holds one of them back.
- A group's credit is computed over the whole group.
- A rollout whose outcome is not `scored` never trains as reward 0.
- A group's statistics land in the step that consumes it, not in the step it settled in.
- A policy publishes at most once a step, as one new version, however many updates the step and its hooks made; with
  no step hook registered, a step's end publishes and advances as it always did (an sft step's publishes nothing).
- Each sft epoch walks every conversation once, in its seeded order, also across a resume.
- No checkpoint saves half a step: an interrupt inside a step that sent training work (when the trainer's steps are
  atomic) or whose end runs its hooks (when they are not) saves nothing of it.
- Under `rollout_isolation: process`, a failure of the first publication of the sampler is fatal: workers would have
  nothing to sample.
- The golden trace ([`tests/train/parity`](../tests/train/parity/README.md)) stays byte-identical unless the change
  means to change what trains.

## Where to start for common changes

| Change | Start at |
|---|---|
| a loss input | [`get_train_data_for_trajectory_collection`](../platoon/utils/tinker_data_processing.py), and the objective's `side_channels` ([`PolicyGradient`](../platoon/train/objectives/policy_gradient.py)) when only the trainer reads it |
| a step metric | [`step/metrics.py`](../platoon/train/step/metrics.py), or [`TrainStep.log`](../platoon/train/step/step.py) |
| a credit rule | [`credit/group.py`](../platoon/train/credit/group.py), and the workflow's [`GroupRolloutWorkflow.compute_baselines`](../platoon/train/workflow/group.py) and [`GroupRolloutWorkflow.assign_advantages`](../platoon/train/workflow/group.py) |
| a task sampler or a group admission rule | [Task samplers](extending/task-samplers.md), [Group admission rules](extending/group-admission.md) |
| a worker setting | [`WorkflowConfig`](../platoon/train/config_defs.py), read through [`GroupSettings`](../platoon/train/workflow/settings.py) |
| what a checkpoint saves | a [`CheckpointParticipant`](../platoon/train/checkpoints/participants.py), or [`RolloutSource.checkpoint_state`](../platoon/train/dispatch/source.py) for the draws |
| a rule that drops stale groups or datums | [`StaleFilter`](../platoon/train/step/staleness.py), and the step's [`TrainStep._filter_stale`](../platoon/train/step/step.py) |
| work at the end of each step | a step hook ([`StepHookBase`](../platoon/train/trainer/hooks.py), [Step hooks](extending/step-hooks.md)) |
| another trainer | [`TrainerBase`](../platoon/train/trainer/api.py) and its hooks, [`TrainerRun.rollouts`](../platoon/train/trainer/run.py) for its draws, and [`TrainerRun.end_step`](../platoon/train/trainer/run.py) to end each step, as the sft trainer ([`SupervisedTrainer`](../platoon/train/trainer/sft.py)) does; [`CORE_TRAINERS`](../platoon/train/trainer/registry.py) to make it a core trainer |
| a dataset format or option of the sft trainer | [`supervised/formats.py`](../platoon/train/supervised/formats.py), [`supervised/registry.py`](../platoon/train/supervised/registry.py) and [`SupervisedOptions`](../platoon/train/supervised/options.py) ([Fine-tune on a dataset](how-to/train-and-eval.md#fine-tune-on-a-dataset)) |

## The tests

`tests/train` mirrors `platoon/train`: the tests of `platoon/train/step` are in `tests/train/step`, and so on; the
task samplers' and admission rules' are in `tests/train/tasks`, and the objective's in `tests/train/policy`. The tests
share one package of fakes, [`tests/train/fakes`](../tests/train/fakes/__init__.py): a run's real config, the rl
trainer on a fake policy and a fake Tinker service, the train loop over groups a test scripts, and the workflow with
scripted members. The sft trainer's tests run it on the fake Tinker service of
[`fake_tinker_service`](../platoon/testing/trainers.py), as the conformance check of every trainer,
[`check_trainer`](../platoon/testing/trainers.py), does. The golden trace,
[`tests/train/parity`](../tests/train/parity/README.md), runs a few steps of each of its scenarios on fakes and
compares everything they did with its golden files.

## Glossary

- **Draw**: a position of the run's order, and the task drawn there, numbered so that a resume can draw it again; it
  is outstanding until a committed step consumes its group.
- **Slot** and **token**: a draw's place in the horizon; a token is a slot a weighted run's adaptive entry will decide
  only when its group can start.
- **Lease**: a draw's rollout permits, which its group's rollouts run on
  ([`DrawLease`](../platoon/train/dispatch/permits.py)).
- **Horizon**: the bound on the groups started and not yet consumed, `(train.max_staleness + 1) × batch_size`.
- **Settled group**: a group that ended, with the entry and label of its row, its record and its draw, as a step
  dequeues it ([`SettledGroup`](../platoon/train/dispatch/settled.py)).
- **Marker**: what settles in place of a group that a group admission rule rejected, or that failed, when it settled.
- **Fate**: what became of a group, in the per-task statistics: `consumed`, `stale`, `failed`, `skipped`, `lost` or
  `rejected`.
- **Verdict**: what a group trains, or why it trains nothing ([`group_verdict`](../platoon/train/credit/group.py)).
- **Kept group**: a group that settled and that no committed step consumed, which a full checkpoint keeps.
- **Stats scope**: the tracker a workflow's statistics go to, `train` or `eval`.
- **Step hook**: work at the end of each step of any trainer, after its own updates and its commit and before it
  publishes ([`StepHook`](../platoon/train/trainer/hooks.py)).
- **Own version**: a datum's policy version under the per-datum stale filter, the oldest its own model calls sampled,
  where every datum of a rollout otherwise carries the rollout's.
- **Lane**: in a run of several trained policies, one policy's way through a step: its rows of each group, its learner
  and its stale filter ([`PolicyLane`](../platoon/train/step/lanes.py)).
- **Share**: when the trained policies step independently, one policy's rows of a settled group, which its own steps
  take ([`Share`](../platoon/train/dispatch/shares.py)).
- **Commit**: when the trained policies step independently, a full save of one policy, which names the row of every
  other policy that a resume from it starts at ([`Commits`](../platoon/train/checkpoints/commits.py)).
