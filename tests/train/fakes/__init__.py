"""The training tests' shared fakes, which a test imports as ``fakes.<module>`` (``tests/train/conftest.py`` puts
``tests/train`` on the path).  A helper that tests of two files share belongs here, not in either test file.

- ``config``: a run's real config: :func:`~fakes.config.make_config` in memory, :func:`~fakes.config.trainer_config`
  from a file with ``key=value`` overrides, :func:`~fakes.config.mixture_config` for a mixture of environments;
- ``launch``: the training and evaluation workflows ``platoon train`` builds for a config
  (:func:`~fakes.launch.built_workflows`);
- ``trainer``: the rl trainer over fake tasks on a fake Tinker service (:func:`~fakes.trainer.make_trainer`), whose
  saves it records (``SaveLog``) and whose logged steps it records (``StepLog``, the ``logged`` of every builder here);
  its workflow, datum and training client; a policy of a run that trains several (:func:`~fakes.trainer.named_policy`);
  a trainer whose steps' saves a test drives itself (:func:`~fakes.trainer.checkpointing_trainer`); and the draws of a
  trainer's last start (:func:`~fakes.trainer.draws_of`);
- ``tinker_service``: a fake Tinker service with tinker-cookbook's own checkpoint saving, and the trainer on it
  (:func:`~fakes.tinker_service.make_service_trainer`), for the runs that save, prune and resume checkpoints;
- ``loop``: the rl trainer's train loop over groups a test scripts, without the start's dispatcher, its full saves
  recorded, not written;
- ``runs``: the runs the dispatch, task and workflow tests drive: groups that say which rows a step trained, the
  training client that records them, one start on the fake service, and what a crash after a checkpoint leaves;
- ``independent``: starts of two policies that step independently on two fake services whose training clients
  remember what each saved state trained (its lineage), for the independent stepping and commit tests;
- ``members``: the group workflow with scripted members, for the tests of the rollout permits, the start limit and
  the startup permit;
- ``dispatcher``: the dispatcher's rows, recorder and samplers, for the dispatcher's own tests;
- ``eval_starts``: the starts of a run with evaluation, crashed where a test says;
- ``pass_rate``: RFC 0019's fake environment of 40 tasks of known pass rates, for ``pass_rate`` and ``informative``;
- ``workers``: a rollout worker's request, and the proxy session of a rollout run in process; ``worker_shutdown``:
  the worker shutdown tests' scenarios;
- ``resume_0_3_0``, ``orders_0_3_0``: the runs and the orders platoon 0.3.0 wrote the fixtures of;
- ``supervised_start_stub``: what ``platoon train`` starts under supervision, in the supervisor's tests;
- ``sandbox_leftovers``: sandboxes a process that is gone left running, for the reaping tests;
- ``worker_stubs/``: the modules a rollout worker or its fork server imports by name in the worker tests, with
  :data:`WORKER_STUBS` on the path (``PYTHONPATH``, or ``sys.path`` in process).
"""

from pathlib import Path

#: The directory of the rollout workers' stub modules, which a worker test puts on the worker's path.
WORKER_STUBS = Path(__file__).resolve().parent / "worker_stubs"
