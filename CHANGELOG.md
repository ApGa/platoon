# Changelog

What each release of platoon changes. Platoon's version is below 1.0: a minor release (0.3.0) may change the Python
API, the config format or the command line, and a patch release (0.2.1) only fixes bugs. To stay on one minor
release, require a range such as `platoon[tinker,ors]>=0.4,<0.5`.

## 0.5.0 (unreleased)

### New features

- **Harbor datasets from the Harbor Hub, git repositories and adapters.** A Harbor environment's `dataset` may name a
  package of the Harbor Hub (`harbor://org/name@ref`, a dataset or a single task, at a revision, a `sha256:` digest
  or a tag), a dataset of Harbor's legacy registry (`harbor://name@version`, from Harbor's `registry.json`, or the
  file or URL `registry` names), or a directory of a git repository (`git+<url>@<commit>#<path>`), and `adapter:
  {repo, commit, path, command, args, env, sparse}` in its place generates the tasks with an upstream adapter at a
  pinned commit, as Frontier-CS's and Harbor's own adapters do. The Hub's packages come through the Harbor CLI at
  platoon's pin (`harbor==0.24.0`, which platoon runs with `uvx` and never imports; `PLATOON_HARBOR_CLI` names
  another), resolved to their digest and downloaded by it, and each task's files must hash to the digest the Hub gives
  it; a registry's tasks and a git directory come through shallow, blobless fetches that check out only the selected
  tasks, and an adapter runs on a checkout of its repository at the commit (whole, or as `sparse` says), its output
  fetched as a local directory. Everything else is as for any fetched dataset: selection, overlays, Harbor's content
  hash, the stack plans of compose files and the stacks they run as (below), the lock, which now also records how the
  source resolved (`source`), and rollout workers that read only the lock and the cache. A Hub revision or digest, a
  full git commit, an adapter's commit and a version in a registry file are pinned, so that `platoon train` and
  `platoon eval` fetch them themselves; a tag, a branch, a registry task at `HEAD` and Harbor's own registry move,
  and `platoon harbor fetch CONFIG --update` resolves them into the lock ([Fetch a dataset by
  name](docs/how-to/sandboxes-and-harbor.md#fetch-a-dataset-by-name)). A symbolic link that leaves its task fails the
  fetch, so that no file of the fetching host reaches the cache, and a URL with credentials is refused; `registry` and
  a local `adapter.repo` are relative to the config file, as every input path is.
- **Harbor compose files are planned when a dataset is fetched, and those that name only `main` run.** A task's
  `environment/docker-compose.yaml` is read as Harbor layers it, with compose's interpolation and Harbor's variables in
  platoon's meaning (`CONTEXT_DIR`, the log directories' `HOST_*_PATH`, `CPUS`, `MEMORY`), into a stack plan that the
  lock records with each task (`compose`, under the lock's `compose_rule`), and its kind in `flags.stack`: `single`,
  `trivial`, the new `main_only`, or `sidecars`. A file that names only `main` and gives it settings runs as the task's
  one sandbox on every provider: its environment, working directory, resources and binds (the task's own files,
  read-only from the cache, a writable bind as a copy; volumes below a bind keep what they mask out of the sandbox), its
  healthcheck, and its command, started as `main`'s init, which the kill before grading spares with what it started;
  an init that is gone by grading makes the attempt an `error` (`main_exited`), and a `copy` fork of such a task is
  refused. A file outside the subset platoon runs, security-relevant keys such as `privileged` or a `cap_add` of
  `NET_ADMIN` included, is refused at fetch, naming the service and key (`compose: <service>.<key>: ...`). A dataset's
  overlay may edit compose services (`[[compose]]`: set, replace or unset their keys, or mark one external) and list
  the host variables a compose file may name (`compose_host_env`), which sessions resolve when they open, so that no
  host value enters a lock; a read-only bind may name the path a listed variable names, and no other host path.
  Stacks with services besides `main` stay refused until platoon runs them (RFC 0010)
  ([Compose files](docs/how-to/sandboxes-and-harbor.md#compose-files)).
- **Sandboxes can keep a task's processes through `kill_processes`.** `Sandbox.keep_processes(entries)`, optional, spares
  processes a session started for the task before any agent ran, with their descendants; the built-in providers have
  it ([Write a sandbox provider](docs/extending/sandbox-providers.md)).
- **Harbor tasks that are multi-container stacks run.** A task whose compose file names services besides `main` (a
  database, a judge, an MCP server) runs as a stack on a sandbox provider with the new `stacks` capability: `main` is
  the sandbox a main-only file would give the task, and each sidecar a sandbox of its own, with its image, command
  (compose's over its image's), environment, healthcheck, volumes (named ones shared
  by the services that name them), binds (writable ones as one copy of the task's files that every service binding it
  shares, which the stack owns), ports and names. The services start in dependency order, under compose's three
  conditions (`service_started`, `service_healthy`, and `service_completed_successfully` for init jobs, each bounded
  by `sandbox_start_budget_s`), and the session opens once all of them are up, as `docker compose up --wait` does, and
  a last probe finds them so; a sidecar that fails, exits or never turns healthy fails the session's start
  (`sidecar_start`, an `error`), and nothing of the stack is left, also when the start is cancelled. Grading runs the
  task's `[[verifier.collect]]` hooks in their services and probes every sidecar before and after the tests (before
  and after the artifacts are collected, for a separate verifier, which also receives the artifacts that
  `artifacts[].service` names from a sidecar's own files): a sidecar that is down makes the outcome `invalid`
  (`sidecar_unhealthy`, with its name, state and log), whatever the reward files say, unless the environment sets
  `on_sidecar_down: grade`. What a sidecar writes in a log directory it binds (`${HOST_VERIFIER_LOGS_PATH}`) reaches the
  tests after grading made that directory afresh, so the agents can neither erase nor forge it. `shared` forks share the
  stack and `copy` forks are refused; a rollout's deadlines count the services' starts and health waits (their images'
  healthchecks too) along their dependencies' longest path; outcomes record each service's run mode and the state
  that the grade's probes found (`services`). On the `process` provider, which isolates nothing, each declared port is
  a port outside the kernel's ephemeral range that the stack holds for its life, which a service learns from
  `PLATOON_PORT_<port>`, and a reference to a service at a port it declares (`scheme://<service>:<port>`,
  `<service>:<port>`) in environment values, commands, healthchecks, collect hooks, `[verifier].env` and the task's MCP
  servers' URLs reaches it, as does a service's own `localhost:<port>`. `stacks: refuse` refuses every stack, and so
  does a provider without the capability (`singularity` and `enroot` where pods cannot run, below); a fetch locks stacks
  as runnable unless its `refuse` sets `sidecars: true` (Changes).
  Stacks that no rollout could start (a dataset directory's registry sidecar that its compose file gives no command, or
  that builds from a context, neither of which a directory's missing image map can give) are skipped by a dataset
  directory and stop a run when a fetched dataset's tasks are listed
  ([Multi-container tasks](docs/how-to/sandboxes-and-harbor.md#multi-container-tasks)).
- **A Harbor environment can run every sandbox without a network.** `network_policy: none` gives every sandbox and
  stack of the environment no network, in every phase, for datasets whose answers the web holds; `task`, the default,
  gives each phase the network its task asks for, and a stack graded in its own `main` gets the verifier's
  ([Multi-container tasks](docs/how-to/sandboxes-and-harbor.md#multi-container-tasks)).
- **Write a provider that runs stacks.** `platoon.sandbox.stacks` defines the stack spec (`StackSpec`, `ServiceSpec`,
  `Healthcheck`, `Dependency`, `Volume`), the running `Stack` (`main`, `services`, `health()`, `logs()`,
  `set_network()`, `connect()`, `rewrite()`, `stop()`), `StackProvider.start_stack()` and the `stacks` capability, and a
  generic driver, `drive_stack`, that starts a stack's services in dependency order over a backend that starts one at a
  time, rolls back on a failure or a cancellation, and probes them
  ([Write a sandbox provider](docs/extending/sandbox-providers.md#run-multi-container-stacks)).
- **Harbor stacks run on Docker, each a `docker compose` project of its own.** The `docker` provider has the `stacks`
  capability: it writes each stack's project file from the stack's spec, as JSON (which compose reads as YAML, so no
  value changes its type), into a directory below its new `stack_root` option, which must be a directory of the
  user's own, with no `:` in its path, and each service is a container that compose creates once the service's
  dependencies hold, running its command (compose's over its image's) as its image's user, which platoon starts in
  order, health-checks and probes as on every provider. The services share an `internal` network of the project's,
  which gives the host no address, so that the host's own services are out of their reach too, and on which each
  answers to its name and aliases; a bridge network is connected for the phases with egress and disconnected for
  those without; `main` reaches the model gateway through a route, as a container without a network does. Volumes are
  directories of the project, filled before any service starts. A start that fails or is given up waits for the
  docker commands it ran, which the daemon carries out whatever becomes of their CLI, and teardown removes the
  containers and networks that carry the stack's label, with the containers' anonymous volumes, then the directory,
  what services wrote there as other users through a container of `main`'s image; a reap by label removes a killed
  worker's stacks, and a stack whose teardown failed stays registered for that. Stacks need the CLI's compose plugin,
  which the provider checks when a run lists its tasks, warning of a daemon older than 25.0.5, whose internal
  networks let DNS queries out ([Stacks on Docker](docs/how-to/sandboxes-and-harbor.md#stacks-on-docker)).
- **`platoon prepare` builds the images of stacks' sidecars.** A sidecar that builds from a context of the task
  (`build:`, planned whole) is built by `platoon prepare` and `platoon harbor images build` with the provider's
  builder, with its build arguments (the `docker` builder's `--build-arg`; the others refuse an image that has any),
  and recorded in the image map, so that no rollout builds one; `platoon harbor images plan` lists it, with the role
  `service`. A run whose sidecars' images the map lacks stops when it lists its tasks, naming the `platoon prepare`
  command, as for the tasks' own images ([Build task images](docs/how-to/sandboxes-and-harbor.md#build-task-images)).
- **Privileges for the stacks of datasets you trust, on Docker alone.** A fetch with `refuse: {privileges: false}` keeps
  the compose services that ask for `privileged`, or for a `cap_add` that a rootless service's root does not hold
  (`NET_ADMIN`; `DAC_READ_SEARCH`, with which a root that is the host's, as on a rootful Docker, reads any of the
  host's files; ...), which a fetch refuses by default, and plans them as the stack's privileges; such a stack runs
  only on a provider with the new `privileged` capability, `docker`, and every other refuses it when a run lists its
  tasks. `platoon.sandbox.privilege_refusal` says why a provider would not give them
  ([Stacks on Docker](docs/how-to/sandboxes-and-harbor.md#stacks-on-docker)).
- **Harbor stacks run as rootless pods on Singularity, and on Apptainer and enroot (experimental).** With hpc-sandbox
  0.5, the `singularity` and `enroot` providers declare the `stacks` capability and start each stack as an hpc-sandbox
  pod: one user and network namespace of its own, whose services share one loopback, so each reaches the others by name
  at the ports it declares, as in compose, and no stack reaches another's. `main` runs as root (your uid, as a single
  sandbox does); a sidecar whose image runs as root runs as root with your subordinate ids, so that the official
  database images' own switch to their user works as in Docker, where the runtime gives that mode (not enroot) and your
  user has them, else as your own uid, as does a sidecar whose image runs as another user. `main`'s upstreams are the
  pod's, which its holder relays, so the gateway's route in `main` is out of the agents' reach; a stack's network
  switches for the whole pod per phase (slirp4netns for egress); its volumes are directories in the pod's own directory,
  with Docker's modes, which the pod's deletion takes, with what a sidecar wrote there as another user. A log directory
  that a sidecar with your subordinate ids binds reaches the tests through that sidecar, which reads what its own users
  keep from yours, and a file grading cannot read makes the outcome an `error`. A sidecar's read-only binds of task
  files that not every user may read (a dataset fetched under a umask such as 027) bind a copy that every user reads, as
  a Docker host's checkout is, since a sidecar may read them as a user of its own; the stack's other copies get those
  modes too. Every connection between a pod's services comes from `127.0.0.1`, so a sidecar that trusts loopback (the
  official postgres image does, for `127.0.0.1`) is open to `main`'s agents, where compose gives `main` an address of
  its own: the docs say so and how a task closes it. Outcomes record each service's run mode (`services`). Stacks are
  refused when the tasks are listed, naming what is missing, with an hpc-sandbox older than 0.5 or on a node that does
  not nest unprivileged user namespaces
  ([Stacks as pods](docs/how-to/sandboxes-and-harbor.md#stacks-as-pods-on-singularity-apptainer-and-enroot)).
- **Harbor stacks on the host network, where pods cannot run (`stacks: host_network`).** On a site whose nodes do not
  nest unprivileged user namespaces, or without hpc-sandbox 0.5, the host-network fallback runs a fetched dataset's
  stacks with each service a single sandbox of the provider (`singularity`, `enroot`, `process`) on the node's own
  network, as the `process` provider runs its stacks: each declared port is a port of the node that the stack holds for
  its life, which the service learns from its listen variables, and the references to the services (above) reach them.
  Since every process of the node reaches a service there, a dataset's overlays say how each one runs safely
  (`host_network = {listen, token, bind}` in a `[[compose]]` entry): the variables that set the port it listens on,
  carry a token that each stack draws anew and gives to its own services alone (through their environment: a fetch
  refuses a rule's variable that the compose file interpolates where the fallback sets it, `$$NAME` passing it on), and
  bind it to `127.0.0.1`. The fallback checks what listens at each port a service declares, `main`'s too: a listener
  beyond the loopback, or one that another user's process holds, makes the service unhealthy. Its stacks' directories
  are in a root of your uid's alone (0700; a symlink or another user's directory there is refused). A fetch scans each
  stack's text files and plan for the references to its services that the fallback could not rewrite
  (`compose.unrewritable_references` in the lock: a service named with a port in `instruction.md`, say), which an
  overlay's `[[edit]]` passes through a variable instead. Refused, when the tasks are listed (exit 78) and when a
  session opens: a sidecar with ports that no overlay authenticates, a declared TCP port without a listen rule, a UDP
  port, such a reference, a task MCP server that is a sidecar (an MCP client sends no token), a stack without a network
  in a phase, and a provider whose sandboxes run other users (`docker`, `fakeroot`); a dataset directory skips every
  stack, saying to fetch it with an overlay. Nothing chooses the fallback by itself (the refusal of stacks under
  `stacks: auto` names it where it could run them), and it needs no pods, so hpc-sandbox 0.4 runs it.
  `platoon.sandbox.host_network_stacks.HostNetworkStackProvider` runs it over any provider whose sandboxes share the
  host's network as your own uid
  ([Stacks on the host network](docs/how-to/sandboxes-and-harbor.md#stacks-on-the-host-network)).
- **`platoon prepare` makes what stacks need.** Besides the sidecars' images (above), it **pins** the registry images
  that sidecars run, on every provider but `process`: the digest each registry serves, with the configuration it gives
  the image for this host's platform (`ENTRYPOINT`, `CMD`, `ENV`, `WORKDIR`, `USER`, `HEALTHCHECK`, ...), which a
  sidecar runs with, since platoon gives every runtime a sidecar's whole command and a pod's runtimes apply none of an
  image's configuration; and, where stacks run as pods with egress, fetches slirp4netns (`hpc-sandbox fetch-tools`)
  when the node has none. A run whose sidecars' images or pins the image map lacks stops when its tasks are listed
  (exit 78), naming `platoon prepare`. `platoon harbor images pin` pins them too, `platoon harbor images plan` lists
  them, and `platoon harbor images verify` checks them
  ([Build task images](docs/how-to/sandboxes-and-harbor.md#build-task-images)).
- **Harbor tasks run as the users they name (`agent_user: honor`).** As Harbor runs them, the agents' commands (a
  harness in the task sandbox, the env tools, `platoon-oracle`) run as `[agent].user`, else the image's `USER`, else
  root; the tests as `[verifier].user`, else the image's `USER`, else root; a `[[verifier.collect]]` hook as its own
  `user`; and platoon's own commands as root, in sandboxes that give root and other users: Docker's (`docker exec
  --user`), and Singularity's with `fakeroot: true` on the copy layout, where platoon switches each command to its user
  with the image's `setpriv`, `su` or Python, with the user's `HOME`, `USER` and `LOGNAME` from the image's
  `/etc/passwd`. Any of Docker's `--user` forms names a user. A session's `sandbox` is the agents' view of the sandbox,
  while grading and the kills keep root's; their commands run with Docker's umask, `/logs/agent` and `/logs/artifacts`
  are every user's (777, as Harbor makes them), and so is `/logs/verifier`, with `/tests` the tests' own, where the
  tests run as a user other than root (tests that run as root get both as root's, `/tests` with the host's modes, as
  Harbor copies it); what platoon uploads for the agents is theirs, a deferred `COPY --chown` gives its owner what it
  copies and a deferred `WORKDIR` the directories it makes to the `USER` in effect there, as BuildKit does (such a
  sandbox starts in `/`, and its commands run in the working directory), and the outcome records the task's users
  (`users`). A `USER` that ends a Dockerfile is the
  agents' user (a fetch still refuses it unless `refuse: {agent_user: false}`), expanded against the image's
  environment where it names it (`USER ${NB_UID}`). Refused, when the tasks are listed (exit 78) and when a session
  opens: users on `enroot`, `process` and Singularity without `fakeroot` or on the fuse-overlay layout, multi-container
  stacks in which users would run (every stack under `agent_user: honor`, compose projects and pods alike: stacks that
  honor users follow), a user the image lacks, an image with no way to switch to one; a `copy` fork on Singularity,
  whose snapshots keep no owners.
  `verifier_user: honor` alone honors `[verifier].user`, now any user, and a root verifier graded in its agents' own
  sandbox runs where the agents run as their user
  ([Users in task sandboxes](docs/how-to/sandboxes-and-harbor.md#users-in-task-sandboxes)).
- **Images keep their file owners where users are honored.** hpc-sandbox extracts every image as you, so a user that an
  image gives a directory would find it root's. `platoon prepare` records each image's owner manifest, the paths its
  layers give users other than root: the `oci` builder lists its OCI-SIF layer's owners before it unpacks it
  (`<key>.owners` beside the image, `file_owners` in the map; `unsquashfs -lln` needs squashfs-tools 4.4, and with an
  older one the build records none and says so), and a task's `docker_image` becomes a *pulled* entry of the image map,
  the image its registry serves pinned by digest, with its `USER` and the owners its layers' tar headers give (whiteouts
  applied), which the sandboxes that honor users run. The record also tells such a sandbox its image's `USER`, which it
  cannot read itself. `SandboxSpec.file_owners` hands a manifest to hpc-sandbox 0.5, whose `fakeroot` sandboxes apply it
  before anything runs; with an older one, such as the 0.4.0 of platoon 0.4, a task whose image gives files to
  users other than root is refused when it is listed, naming 0.5. A task whose sandbox runs a user other than root, or
  anything as its image's `USER` (the agents' under `agent_user: honor`), is refused where the image has no record:
  built by platoon before 0.5, the `def` builder (Apptainer's) or the `oci` builder with a squashfs-tools before 4.4
  (`platoon prepare` names the build), a `docker_image` that `platoon prepare` has not recorded, a local image file, or
  a dataset directory's, which has no image map. A sandbox in which only root runs needs none. Docker keeps its images'
  owners itself ([Users in task sandboxes](docs/how-to/sandboxes-and-harbor.md#users-in-task-sandboxes)).
- **Sandboxes can run a command as another user.** `Sandbox.exec_as(user, cmd, ...)`, optional, runs one as a user of
  the image (the docker provider's); a provider whose sandboxes lose their images' file owners says so
  (`loses_file_owners`) and whether it can apply a manifest (`file_owners_refusal(options)`;
  `platoon.sandbox.loses_file_owners`, `file_owners_refusal`)
  ([Write a sandbox provider](docs/extending/sandbox-providers.md)).
- **A Harbor stack's MCP servers are env tools.** Each `sse` or `streamable-http` server of a task's
  `[[environment.mcp_servers]]` whose `http://` URL names a service of its stack (`http://mcp-server:8000/mcp`) serves
  its tools through the tool plane, as the session's env tools, each named for the server and the tool in ASCII letters,
  digits and underscores (`mcp_server__get_secret`): read-only where the tool's `readOnlyHint` says so, filtered by the
  agent's access level, recorded with the agent's tool calls, called in their turn with the generic tools within
  `tool_timeout_s`, and their text cut as the generic tools' output is. platoon's own MCP client speaks both transports,
  streamable HTTP and the legacy SSE transport, Harbor's default for task servers, with the protocol versions the
  gateway serves, reads no answer or listing longer than 16 MiB, follows a server's redirects within its origin alone,
  sends each request on a connection of its own (one that a server's keep-alive ends as a call arrives would lose it),
  and brings no `mcp` package into the rollout workers. The session lists the tools as it opens: a server that cannot be
  reached or does not list its tools fails its start (`sidecar_start`), naming it, rather than run the rollout without
  them ([Task MCP servers](docs/how-to/sandboxes-and-harbor.md#task-mcp-servers)).
- **This host reaches a stack's services on every stack provider.** `Stack.connect(service, port)` gives the docker
  provider's stacks a relay: a container in the network of the service that listens there, running platoon's forwarder
  with the Python 3 of that service's image, else of `main`'s, which serves a Unix socket in the project's directory,
  whatever the stack's networks, also from a rootless daemon; a pod's port is the socket that hpc-sandbox 0.5 exposes;
  the `process` provider's is the port of the host that the stack holds. A socket path too long for an address is
  handed out as a link in node scratch (`platoon.sandbox.stacks.SocketLinks`)
  ([Write a sandbox provider](docs/extending/sandbox-providers.md#run-multi-container-stacks)).
- **Harness specs get a task's own MCP servers.** `{task_mcp_servers}` renders the MCP servers a task offers its agents,
  as a JSON list of each one's `name`, `transport` and `url` as the task's sandbox reaches it (or a `stdio` server's
  `command` and `args`), for a harness in the task's sandbox that reaches them itself, as Harbor's agents do; a harness
  that runs locally gets `[]`, and reaches only the servers that the env's tools serve. A session gives them as
  `task_mcp_servers`, an optional attribute ([Placeholders](docs/tutorials/cli-harness.md#placeholders)).

- **One lifecycle for every trainer, and the trainer API (experimental).** Each start of training goes through
  `platoon.train.trainer.run.TrainerRun`, in the order a start always ran: the trainer's checks, the signal
  handlers, the policies' service clients and the start point, their training clients, the trainer's preparation,
  the launch save, the trainer's readying, its steps, the save at exit and its closing; a trainer's part of each phase
  is a hook of `platoon.experimental.TrainerBase` (`check`, `prepare`, `started`, `ready`, `run_steps`, `close`),
  and `PlatoonTinkerRLTrainer` is one. `platoon.experimental` exports `Trainer`, `TrainerBase`, `TrainerSetup`,
  `TrainerNeeds`, `TaskSources`, `Workflows`, `StepPlan`, `PolicySpec`, `MAIN`,
  `CheckpointParticipant` (a part of the run's state that every full checkpoint holds, after the run's counters, in
  the order the parts registered) and `Objective`, and `platoon.testing.check_trainer` checks a trainer: it trains a
  run of two steps on a fake Tinker service, stops one at a step boundary and resumes it. The config's `trainer:`
  selects the trainer: `rl`, the default, or `sft` (below; `interleaved` is reserved). Nothing a run does changes.
- **Supervised fine-tuning on LLaMA-Factory datasets (`trainer: sft`).** A run fine-tunes its policy on chat datasets
  instead of rollouts, with the loss on the model's turns: the mean negative log-likelihood of the assistant's
  replies and tool calls (with `mask_history`, of each conversation's last one only), rendered by the policy's renderer
  as the policy sees them when it samples. The datasets follow LLaMA-Factory's conventions: a `dataset_info.json` in
  `trainer_options.dataset_dir` names each one, on the Hugging Face Hub (`hf_hub_url`) or in files (`file_name`:
  json, jsonl, csv, parquet or arrow), with its `formatting`, `split`, `subset`, `folder`, `num_samples`, `columns`
  and `tags`; `alpaca` rows (`instruction`, `input`, `output`, `system`, `history`) and `sharegpt` conversations
  (`from`/`value` or `role`/`content` messages, a system prompt, `tools`, and `function_call` and `observation`
  turns) become conversations as LLaMA-Factory's converters align them, and the rows it would skip are skipped.
  An entry's `columns` and `tags` replace LLaMA-Factory's defaults whole, as they do there. `trainer_options` sets the
  datasets, `dataset_dir` (relative to the config file; left out, `data` beside it), `batch_size`, `num_epochs` or
  `max_training_steps`, `num_microbatches`, `max_length` (shared among a conversation's turns as LLaMA-Factory's
  `cutoff_len` is), `mask_history` and `seed`, checked when the config loads, `dataset_info.json` with them; `train:`
  declares the policy, and its keys that only `rl` reads are refused under `sft` when set, naming the option to set
  instead. Before the service opens, the run renders the first conversation, the first with tools, the first with a
  tool call and the first with neither a system prompt nor tools, and refuses a renderer that cannot render one,
  naming its dataset and row (a system message a renderer adds of its own does not train). Each epoch walks every
  conversation once in a seeded order, and every full checkpoint saves where the run is (`loop_state.sft`), so a
  resume trains each conversation once an epoch, also with another `batch_size`; a resume with other conversations or
  another seed is refused. `checkpoint.keep_recovery` needs no `train.max_staleness` under `sft`, since nothing
  samples the policy. `platoon config validate --tasks` loads the datasets of an sft config. Preference and
  multimodal datasets, and evaluation during training, are not there yet
  ([Fine-tune on a dataset](docs/how-to/train-and-eval.md#fine-tune-on-a-dataset), which lists where it differs from
  LLaMA-Factory; [`examples/sft`](examples/sft/README.md)). Its steps are atomic and its updates fail closed, and
  each step ends with `run.end_step(sampled=False)`: the step hooks a subclass registers run there, and since nothing
  else samples the policy, a step without a hook, or one that updated nothing, publishes no sampler and saves only the
  full checkpoint the cadence asks for. The trainer API gains what such a trainer needs: that `sampled=False`,
  `TrainerBase.stats_logger` and `train_tracker`; a trainable policy has its optimizer's parameters once it opens, the
  run stops the clock's watchdog after `run_steps`, and a recovery checkpoint that a step's end saves goes to the
  trainer's pruning.
  `platoon.testing.check_trainer` takes `options_for_steps`, the options of a run of so many steps, for a trainer
  whose options say how long it trains, and `platoon.testing.fake_tinker_service` is the fake service it runs on, for
  a test of a run of one's own: it renders with the toy renderer and records the datums it trains on and the
  checkpoints it deletes.

- **Any trainer draws the run's rollouts, and the workflow contract (experimental).** A trainer whose
  `needs()` say it rolls out gets the start's draws from its run, `run.rollouts()`
  (`platoon.train.dispatch.source.RolloutSource`): the run's task order and samplers, the group admission rules, the
  per-task statistics, the dispatcher with its rollout permits and exact resume, and the groups they settle into
  (`take`, which ends on the run's shutdown request, `committed` and `step_started`), which take part in every full
  checkpoint; the run builds its workflows from its setup.
  `platoon.experimental` exports the workflow contract: `RolloutWorkflow`, `WorkflowCapabilities`, `capabilities_of`,
  `WorkflowSetup`, `RolloutOutput`, `OutputReport`, `report_of` and `rows_by_policy`. `platoon train` builds the run's
  workflow class from a `WorkflowSetup`, with its `from_setup` when it has one (`GroupRolloutWorkflow` does), else with
  `GroupRolloutWorkflow`'s keyword arguments, as before. Nothing a run does changes.

- **Step hooks: work at the end of each training step (experimental).** A step hook (`platoon.experimental.StepHook`,
  `StepHookBase`) runs at the end of every step of any trainer, after the step's own updates and its commit and before
  it publishes: each hook drains its side work (`drain`), adds optimizer updates of the step's policy on the same
  training client and Adam state (`update`, through `end.train(policy, datums, objective=...)` or
  `end.learner(policy, objective=...)`), and hears what the step published (`published`, also when nothing did, and
  once after the launch). Platoon counts the hooks' updates from the policy's optimizer steps, and the step publishes
  each policy it or a hook updated once, as one new version, so the hooks spend no staleness budget; the step's row
  gains `batch/hook_optimizer_updates`, and each hook's metrics go under `<name>/`. An update takes Adam parameters of
  its own (`PolicyUpdate.apply(adam_params=...)`, `end.train(..., adam_params=...)`). A hook alone is tested on
  `StepEnd.for_test(learner=...)` ([Write a step hook](docs/extending/step-hooks.md), `examples/replay-sft-hook`).
- **Attaching step hooks.** The rl trainer attaches the hooks its options name, from the config:
  `trainer_options: {hooks: [pkg.mod:Hook, {hook: pkg.mod:Other, options: {...}}]}`, checked when the config loads and
  built with the class's `from_run(run, options)` before the training service opens; a hook that does not import,
  build or register stops the run with exit code 78. A trainer of its own registers its hooks in `check()`
  (`run.hooks.register`, which also makes a hook with `keys` a participant of the run's checkpoints) and ends each step
  with `run.end_step()`: the hooks, then each policy that the step's updates through `run.learner` or a hook's moved
  publishes once (`run.publish`: a full checkpoint on the cadence, else a weights-only save), and the rollout workers
  of the draws the run opened get each new sampler, the launch's first.
- **Failures and interrupts around step hooks.** A hook that raises in `drain` or `update` fails the start before
  anything publishes; one that is not `critical` only hears publications, and its errors are logged. A hook's learner
  sends and counts during the `update()` that took it only, under another objective too, the trainer's `run.learner`
  sends nothing while the hooks run, a policy takes one update at a time, and a step's end refuses an update left open
  (sent, not applied), with hooks or without. A hook's `end.wait(..., interruptible=True)` raises
  `TrainInterrupted` (now in `platoon.experimental`) on a stop request when the trainer's steps are atomic; when they
  are not, the rl trainer's, the step ends first, and an interrupt that comes as the hooks run all the same saves no
  checkpoint of the step, so that none saves half a step.
- **The rules of a run's checkpoints and close.** A participant's keys are its own: `run.checkpoints.register` refuses
  the counters', those kept for several policies, the fields that a `checkpoints.jsonl` row holds itself (`name`,
  `epoch`, `final`, `state_path` and `sampler_path`, read from tinker_cookbook's `CheckpointRecord`, which an entry of
  the same name would replace or be replaced by), a step hook's of the draws (`tasks`, `next_position`, `dispatch`,
  `credit`) and another participant's, with a `ConfigError` (exit 78) that names the participant and the key. Those
  registered in `check()` take up the start point before any training client opens, and a key of the resumed row that
  nothing claims stops the start (exit 78). The trainer's `close` now runs in `finally`, also after a failure
  (`run.ended` tells them apart), then each hook's and participant's `aclose`, in reverse order of registration; a start
  that failed raises its error as it was, the close's errors noted on it.
- **What a trainer asks of the run (rule 10).** `TrainerNeeds` gains `atomic_steps` (an interrupt inside a step that
  sent training work saves no `<step>_interrupted` checkpoint of it, and the resume starts from the last full one),
  `fail_closed_updates` (an optimizer step that the service skipped, or whose metrics are not finite, raises
  `UpdateRejected` before it counts) and `draw_provenance` (each training episode sees its draw as `ctx.draw`, a
  `platoon.experimental.DrawInfo`, which a step's end names in `StepEnd.consumed`). The rl trainer keeps the first two
  off, as before.
- **Side inference and warm starts.** `Policy.snapshot()` (`platoon.experimental.PolicySnapshot`) samples a policy's
  published sampler and refuses a sample once the policy has published again (`PolicyMoved`); `Policy.renderer` and
  `.tokenizer` are its model's. `checkpoint.load_optimizer_state: true` starts a new run from `load_checkpoint_path`'s
  weights and its optimizer state. `platoon.testing.check_trainer` runs a step hook of its own at each step's end
  (`runs step hooks`) and stops a run as the hooks drain a step (`keeps steps whole`). Nothing a run without hooks does
  changes.
- **Per-datum stale masks (experimental).** `train.stale_filter: datum` drops only the datums of a group that are more
  policy versions old than its bound, where `group`, the default, drops the whole group by its oldest datum: a rollout
  that spans a published policy keeps the datums it sampled since, and the others keep the advantages their group's
  credit gave them. Each model call records the policy version it sampled (`TinkerLLMInteraction.version` and
  `ChatResponse.version`, defaulted; a rollout worker's `FollowingSamplingClient` pins the client and version of each
  call together), each datum is judged by the oldest among its own calls, and a group none of whose datums is within
  the bound is dropped whole, as before. `train/batch/stale_datums` counts the datums dropped. A workflow takes the
  setting through the new `takes_stale_filter` capability and its `stale_filter`, as `GroupRolloutWorkflow` does
  (RFC 0020's P5; [Per-datum stale masks](docs/how-to/train-and-eval.md#per-datum-stale-masks-experimental)). Under
  `group` nothing a run does changes, but the staleness warning's last advice, which named the masks as planned, now
  names `train.stale_filter: datum`.
- **Credit estimators, chosen per environment entry (experimental).** `environments[].credit` (with `credit_options`)
  chooses how an entry's training groups turn rewards into advantages, in place of the workflow's own credit:
  `group` (that credit, as an estimator, value for value), `keyed` (each agent's rows centered on the mean reward of
  the group's agents that share its key, any of `task`, `policy`, `role` and `depth`, optionally leave-one-out: under
  `[task, role]` a subagent is centered on the group's subagents, not on its rollout's root baseline), a name an entry's
  `package` registers with `@credit_estimator`, or `module:Class`. The training workflow awaits the entry's estimator
  once per group, in the trainer, after every member has settled; evaluation never runs one. An estimator that reads
  agents (`needs_agents`) has the rollout workers tag each row with the agent that produced it (worker protocol 3,
  `PolicyRolloutWorkerRequest`, sent only then), and sees every agent of every member with its role, policy, depth,
  parent, own reward and rows; the group's output is then `PolicyGroupOutput`, its rows with their tags, which a full
  checkpoint keeps whole. An estimator with state changes it in `settle(group)` as the run files each group, and each
  full checkpoint saves it (`loop_state.credit`) with the groups it keeps. An estimator that raises, or whose `assign`
  does not return within its `assign_timeout_seconds` (600 by default), stops the run with a `CreditEstimatorError`
  that names it and its entry. Loading resolves each estimator, checks its
  options and runs its `check` (exit 78), refuses one beside a workflow that does not apply it (the new capability
  `applies_credit`; a `GroupRolloutWorkflow` subclass that overrides `compute_baselines` or `assign_advantages` keeps
  crediting with them), and `lock.json` records it (`kind: credit`). `platoon.experimental` exports the credit API
  (`CreditEstimator`, `CreditEstimatorBase`, `CreditSetup`, `CreditCheckContext`, `CreditGroup`, `MemberRollout`,
  `AgentRun`, `TrainingSample`, `PolicyView`, `credit_estimator`), and `platoon.testing.check_credit_estimator` checks
  an estimator: advantages on action tokens only, and a resume that credits as the run would have
  ([Credit each environment's groups](docs/how-to/train-and-eval.md#credit-each-environments-groups-experimental),
  [Write a credit estimator](docs/extending/credit-estimators.md);
  [`examples/credit-by-role`](examples/credit-by-role/README.md) gives each role of a delegating rollout a baseline of
  its own). A run whose entries set no `credit` runs as before, request for request.

- **Several policies in one run: frozen policies that agents sample or a credit estimator scores with
  (experimental).** `policies:` declares the run's other policies under `train:`'s names (`model_name`,
  `renderer_name`, `renderer_kwargs`, `context_window_length`, `sample_timeout_seconds`, `tinker_base_url`,
  `load_checkpoint_path`) and `trainable`; `train:` still declares `main`, the policy the run trains, as before. This
  release trains `main` only: every other policy says `trainable: false` and is frozen, sampled or scoring and never
  stepped, saved or pruned; it opens only the sampling client it needs, of its base model or of the `tinker://`
  sampler its `load_checkpoint_path` names. A policy without `tinker_base_url` uses the run's service; each policy
  opens a Tinker session of its own, also on a shared service, so that a fatal error on one policy's session never
  ends another's, and an error opening a policy other than `main` names it and its service. `agents:` binds agent
  roles to policies (`agents.roles.<role>.policy`): the root agent samples `main`, and an agent opened with a role
  (`ctx.agents.spawn(role=...)`, `POST /v1/platoon/agents`, a program's judge) samples its role's policy for its whole
  life, through the gateway's runtime of that policy, with its own renderer and tokenizer
  (`ModelGateway.serve_policies`); each of its interactions names that policy (`TinkerLLMInteraction.policy`). A
  frozen policy's agents make no rows, but their rewards and tags reach the credit, each row is stamped with its own
  policy's version, and rollout workers follow one binding file per sampled policy (`model_binding.<id>.pkl`, worker
  protocol 3), which the trainer publishes before the first rollout and each evaluation (the workflow capability
  `binds_policies`, which `GroupRolloutWorkflow` declares). `identity.json` records each policy's model, renderer and
  training, a frozen one's sampler, and each role's policy, never a URL, and a resume that changes them stops with
  exit code 78, naming each change. Loading, and `platoon config validate`, refuse an id that is no lowercase name,
  `policies.main`, a second trainable policy (training several arrives with lockstep multi-policy training, RFC 0021's
  milestone B4), a policy's `sample_timeout_seconds` that is not a positive number (as `train:`'s), a role bound to
  an undeclared policy, `agents.policy` other than `main`, roles bound to several policies beside a workflow that
  binds one, and either key under `trainer: sft`, which rolls nothing out, and warn of a frozen policy nothing uses
  ([Sample several policies](docs/how-to/train-and-eval.md#sample-several-policies-experimental)).
  `platoon.experimental` exports `AgentBindings`, and `TrainerSetup` takes `agents`. A run without `policies:` or
  `agents:` runs as before, request for request.
- **On-policy distillation from a frozen teacher (experimental): `credit: reverse_kl`.** For each training group, a
  frozen teacher (`credit_options.teacher`, a policy of `policies:`) scores every token the student sampled
  (`compute_logprobs`), and each action token's advantage is `-kl_coef * (student logprob - teacher logprob)`, plus
  `reward_coef` times the row's reward centered on its member's baseline (0 by default); with `reward_coef: 1` this is
  tinker-cookbook's on-policy distillation advantage, without its discounted KL. The teacher's calls run in the
  trainer after the group's rollouts returned their permits, at most 32 in flight, each bounded by the teacher's
  `sample_timeout_seconds`; one that fails or runs out of time, or a response of the wrong length or without a
  target token's logprob, fails its group alone (verdict `failed`, counted under `credit/failures`), which trains
  nothing, and the run goes on. An evaluation never calls the teacher, and the step logs `credit/reverse_kl/avg`.
  Loading, and `platoon config validate`, refuse a teacher that is no frozen policy of `policies:`, and one whose
  tokenizer differs from the student's (exit 78, before any Tinker session opens; when a tokenizer cannot be loaded
  then, `platoon train` compares the ones it loaded for the two policies as it builds the estimator). A credit
  estimator reads the run's policies from `CreditSetup.policy_views` and `CreditGroup.policies` (`PolicyView.score`:
  a policy's logprobs of each datum's target tokens, which raises `ScoringFailed` to fail the group; its `tokenizer`
  and `version`) ([Distill a frozen
  teacher](docs/how-to/train-and-eval.md#distill-a-frozen-teacher-experimental);
  [`examples/on-policy-distillation`](examples/on-policy-distillation/README.md) distills Qwen3-32B into the
  quickstart's Qwen3-4B, the teacher on the student's service and on its own).
- **A tool allowlist per agent role (experimental): `agents.roles.<role>.tools`.** A role's list of tool names or
  `fnmatch` patterns (`read_*`; a Harbor task's MCP tools by the names agents see, `<server>__<tool>`) limits the agents
  of that role, and every agent under them, to those tools: a child spawned or minted with the role, a program's `judge`
  or `verifier` (which its spawner's lists limit, not those of the child it judges), and the root agent when the list is
  under its role, `policy`. Their listings of env tools (`ctx.list_tools()`, MCP `tools/list`) hold only those, and the
  gateway refuses a call to any other where it runs the call, for inline and MCP harnesses alike: the agent gets an
  error result that says why, the environment never sees the call, the run goes on, and the agent's trajectory counts
  the refusals in its misc (`tools_refused: {count, first}`). A role without `tools` keeps every tool, and `tools: []`
  allows none; a role may give `tools` without a `policy` (`agents.roles.<role>.policy` is now optional, its agents then
  sampling `agents.policy`'s). Rollout workers get the lists in their requests (worker protocol 3's `role_tools`), and
  in-process rollouts from the trainer's gateway (`ModelGateway.limit_tools`). Loading, and `platoon config validate`,
  refuse `tools` that is no list of non-empty names (exit 78), warn of a role that looks like a misspelling of one of
  platoon's own (`subagents` for `subagent`), and of a pattern that matches no tool of the run's environments when each
  declares its tools up front: an ORS environment run in process whose class lists them all, a task table, or a service
  class with `declared_tools` ([Limit the tools each role may
  call](docs/how-to/train-and-eval.md#limit-the-tools-each-role-may-call-experimental);
  [`examples/subagents/read-only-subagents.yaml`](examples/subagents/read-only-subagents.yaml) trains the subagents
  example with subagents that may only read). A run without `tools:` runs as before, request for request.

### Changes

- A `process` sandbox whose working directory is `/` runs in its own `/tmp`, since no path maps to the host's root; the
  provider refused such a spec.
- The `process` provider's root (`platoon-sandboxes-<uid>` in node scratch) is a directory of your uid's alone, as the
  `docker` provider's `stack_root` is: made 0700, and refused, with nothing made there or reaped from it, when it is a
  symlink or another user's directory, which another user of a node's shared scratch could have put there.
- A `singularity` or `enroot` sandbox whose start is given up (a rollout cancelled while its sandbox starts) is stopped
  once its start is done, which then runs to its end: cancelled while Singularity made its instance, hpc-sandbox's own
  rollback could fail and leave the instance running until the worker exited.
- Harbor reads only `environment/docker-compose.yaml`, and so does platoon now: a task whose compose file is named
  `compose.yaml`, `compose.yml` or `docker-compose.yml` runs as one container without it, as in Harbor, where platoon
  refused it (or ran it, when it only named `main`), and the fetch warns of it (`flags.unread_compose`); a task with a
  prebuilt image, no Dockerfile and such a file gets its `environment/` uploaded, as Harbor does.
- A lock records each task's stack plan. A dataset pinned to a commit whose lock platoon 0.4.0 wrote keeps answering
  its config where the lock refused no task as a stack (`stack`), and also, with its main-only tasks refused as 0.4.0
  refused them, where the config sets `refuse: {sidecars: true}`: fetch it again to run them (`platoon harbor fetch
  CONFIG --repair`). Where it refused stacks and the config sets no `refuse.sidecars`, the dataset is fetched again
  (below). A lock whose plans follow another compose rule (`compose_rule`) is not read: fetch the dataset again.
- **A fetch locks multi-container tasks as runnable stacks** (`refuse.sidecars` defaults to `false`), where platoon 0.4
  refused them by default (`stack`), and with `sidecars: false` locked them refused (`task_config`). Where the
  provider runs no stacks, a run stops when it lists its tasks, naming them: leave them out (`exclude_tasks`), or set
  `refuse: {sidecars: true}`, which refuses them as before, as `stack: the compose file adds sidecar services (refuse:
  {sidecars: false} fetches it as a stack)`. A lock written for an entry that sets no `refuse.sidecars` and that
  refused stacks no longer answers it: `platoon prepare`, `platoon train` and `platoon eval` fetch a pinned dataset
  again (its cached tasks are reused), and a rollout worker that finds no lock names the old one; a lock that refused
  none still answers. A dataset directory's multi-container tasks are listed where the provider runs stacks, and
  skipped with a warning, as before, where it runs none.
- The `hpc` extra requires hpc-sandbox 0.5.1 (`hpc-sandbox>=0.5.1,<0.6`), where platoon 0.4 took 0.4 (`>=0.4.0,<0.5`):
  0.5 has the pods that stacks run as and applies owner manifests, and 0.5.1 fixes pods for clusters: a sidecar that
  switches users starts where the run's directory or the task's files are on storage that only one of your groups opens,
  and reaches its binds under a umask such as 027, and a pod's `main` outlives an agent's `kill -9 -1`, which with 0.5.0
  ended it and made the attempt an `error`. With an older hpc-sandbox, which an environment that installs platoon
  without the extra may keep, stacks, and the tasks whose sandboxes need an owner manifest, are still refused when the
  tasks are listed, naming 0.5 ([The hpc extra](docs/getting-started/install.md#the-hpc-extra)).
- The `def` builder's images record their base image's configuration (`ENTRYPOINT`, `CMD`, `USER`, `HEALTHCHECK`,
  `EXPOSE`, ...), from its registry, under what the Dockerfile sets, since a definition file keeps none of it (a
  sidecar's image, whose command may be its base's, fails to build without it); and the `singularity` provider's
  images are built with `def` where its runtime is Apptainer, which has no BuildKit for `oci`.
- `[[verifier.collect]]` hooks, which platoon refused, run: in `main` for a single-container task, in their service for
  a stack, `main`'s first, best effort (`collect_failed` in the outcome). A hook that names a `user` is refused unless
  the users policy ignores it: `verifier_user: ignore`, or, where `verifier_user` is unset, `agent_user: ignore`.
- The capabilities a provider may declare (`platoon.sandbox.base.CAPABILITIES`) include `stacks`, which the `process`
  provider declares: a provider that copies another's capabilities, as the provider guide's wrapper did, declares
  `stacks` only with a `start_stack` of its own. The `process` provider links its mounts before it makes the working
  directory, so that a working directory that is a mount's target is that mount, and never makes it inside a read-only
  mount: a read-only bind of a task's files at the working directory, which it dropped, applies, and since the
  provider enforces no read-only mount, what is written there reaches the task's files.
- Image map entries record an image's `HEALTHCHECK`, `EXPOSE` ports, `VOLUME`s and `STOPSIGNAL` too, and a
  Dockerfile's build plan what its deferred suffix sets in the image's configuration (`config`: `CMD`, `ENTRYPOINT`,
  `HEALTHCHECK`, ...), which a main-only task's init and healthcheck use. The build keys of task images do not change.
- The `docker` provider removes a sandbox's anonymous volumes with it (`docker rm --volumes`), which those of an
  image's `VOLUME`s outlived, also when it reaps; its `reap` removes the labelled stacks first, and counts them with
  the containers. Its sandboxes are reaped through a record that names its `stack_root` too.
- `platoon.sandbox.stack_refusal` asks a built provider's class too (its `stack_refusal(options)`, with the provider's
  `options`), as it asks a provider named by its name.
- A dataset directory's stack whose sidecar builds from a context is refused saying that a fetched dataset's image map
  can hold that image (serve the directory as a dataset to fetch). `examples/harbor-local`'s `needs-a-database` is
  still refused on Docker, which runs stacks now, for its `postgres:16` sidecar, which runs its image's own command,
  which only a fetched dataset's image map records (`platoon prepare` pins it).
- A deferred `COPY` keeps its `--chown` in a lock's build plan (`["COPY", "--chown=<owner>", source, ...,
  destination]`), and a deferred `USER` its place among the deferred steps (`["USER", user]`), which a platoon before
  0.5 does not read: fetch such a dataset again with it. A lock fetched before keeps its copies, and the directories
  that a `WORKDIR` after a `USER` makes, root's where users are honored, and a run that honors users warns of it.
- A root verifier's sandbox (`verifier_user: honor`) gives every user the read and search access to the files of an
  image's deferred `COPY` steps that their group has, as a checkout under Docker's umask gives it, where it kept the
  host's modes (a file of mode 600 stays so); `tests/` keeps the host's modes there, and `/logs/verifier` stays
  root's, as before.
- The docker provider runs its own commands in a container, a compose service's too, as root (`docker exec --user
  0`), whatever the image's `USER`: the kills before grading, the listing of the container's processes, the
  directories an upload makes and the look at what a download copies. Commands keep running as the container's user.
- The `edit` tool reads a file through the sandbox's shell, as the sandbox's commands run, rather than with a download,
  which runs as the sandbox's root; a file larger than a command's output holds (8 MiB on `docker` and `process`, 16
  MiB on hpc-sandbox, in base64) is an error rather than an edit, and so is one that is not a regular file (a FIFO, a
  device), at once, and a read that takes longer than `tool_timeout_s`.
- Staged harness toolchains are readable by every user, their directories and programs runnable by them, so that the
  users a sandbox switches to run them.
- The refusals of tasks that name users read `... and the environment refuses tasks that name users; the task is
  refused unless it sets agent_user: honor, ... or ignore, ...` when a task loads, and `agent_user: the task sets
  [agent].user or [verifier].user (refuse: {agent_user: false} fetches it, ...)` in a lock. `HarborEnvironmentService
  .root_images_closed()` is `closed_images()`, which covers every sandbox that honors users.
- **The training code lives in packages named after the stages of a run.** `platoon/train/tinker/` held the trainer,
  its config, the rollout workflows and the rollout workers; they are now packages of `platoon.train`: `trainer` (one
  start of training: the trainer, the training loop, signals and the watchdog), `policy` (the model training samples
  and steps: its spec, its state, its training client, its own saves and its learner, through which a step updates
  it), `objectives` (what its rows train under: the loss, the loss inputs only the trainer reads and the training
  metrics), `step` (one training step: its metrics, the staleness warning, the batch transforms), `dispatch` (what
  starts the training groups: the dispatcher, the rollout permits, running and settling a group), `workflow` (one group
  of rollouts: `GroupRolloutWorkflow` and its parts), `credit` (a group's baselines and error-token filter), `workers`
  (process-per-rollout), `checkpoints` (what a full checkpoint saves, and resuming) and `evaluation` (evaluation during
  training); the config schema is `platoon.train.config_defs`, and `platoon train`'s start `platoon.train.launch`.
  Nothing a run does changes: the same datums train in the same order, and the metrics, config keys, messages,
  checkpoints and the files a resume reads are as before. `platoon.train.tinker` keeps the policy client (`proxy`), its
  renderers, `key_refusal` and `rollout_metrics`, until a later release moves them, and the shims of the old paths
  (below).
- **The trainer reads the model it trains from its config once, as it is built.** Its policy (`platoon.train.policy`)
  takes the model, the renderer, `tinker_base_url`, `train.lora_rank`, `train.optimizer`, `train.loss_fn`,
  `train.loss_fn_config` and `checkpoint.load_checkpoint_path` from the config when `PlatoonTinkerRLTrainer` is made,
  where it registered its model gateway before: setting one of them in the config after that and before `train()` no
  longer reaches its training client, its updates and their `optim/lr` metric, or its recovery pruner.
  `TrainLoopSharedState` holds the run's policies (`policies`), and `model_info`, `service_client`, `training_client`,
  `sampling_client`, `optimizer_params` and `optimizer_steps` are properties of the policy it trains, so
  `dataclasses.fields()` lists `policies` in their place. It is made with keyword arguments only, and still takes
  those six instead of `policies`, but not beside it: given both, it raises `TypeError`, and so does
  `dataclasses.replace()` of it. A `copy.copy()` of it shares the original's policies.
- `TrainLoopSharedState` holds the run's step clock (`clock`), and `train_step` and `sampling_client_updated_event`
  are properties of it, so `dataclasses.fields()` lists `clock` in their place; it still takes those two instead of
  `clock`, but not beside it. `PlatoonTinkerRLTrainer` saves its launch, interrupted and final checkpoints through the
  run's checkpoints, whose lines (`Saving recovery checkpoint ... at launch`, `Saving permanent checkpoint ...`,
  `Training complete successfully!`) come from `platoon.train.trainer.run`, and its `run_log_path`, `async with` and
  `_await_with_heartbeat` are `TrainerBase`'s, whose lines (`This run labels its sandboxes ...`) come from
  `platoon.train.trainer.api`.
- The rl trainer's draws are a rollout source too, which the run opens, starts and closes as it does any trainer's:
  `PlatoonTinkerRLTrainer` keeps `_recorder`, `_dispatcher` and `_admissions` as properties of its source, its steps
  commit their groups and move the horizon on through the source, and the source's steps of opening, saving and taking
  up the draws go through the trainer's methods of their names (`_begin_task_records`, `_training_tasks`,
  `_new_dispatcher`, `_task_records_started`, `_checkpoint_state`, `_restore_dispatch`; `_dispatch` starts them).
  Those call the source's parts, in `platoon.train.dispatch.source`, `.setup`, `.runner` and
  `platoon.train.checkpoints.saving`, `.resume`, which take the source in place of the trainer, and call what they need
  as module functions, no longer through the trainer: replacing its `_samplers`, `_run_group` or
  `_await_rollout_with_heartbeat` changes no training group (`_run_group` still runs an evaluation's groups), and the
  trainer has no `_run_training_group`, `_settle_under_rule`, `_read_kept_groups`, `_saved_samplings` or
  `_resume_position` any more. The lines of the per-task statistics' start (`This start keeps no per-task statistics
  ...`) come from `platoon.train.dispatch.source`.
- `platoon.train.components` no longer defines `DatasetLoader`, `TaskLoader`, `RolloutFn`, `RewardProcessor`,
  `WorkflowFactory`, `TrainerConfigClass` and `LossFn`, which nothing used. `ClientThroughputMonitor` no longer has
  `record_dequeue_to_first_model_call`, which nothing called.
- `GroupRolloutWorkflow` declares at class level the defaults of what its constructor sets (`stats_scope`,
  `rollout_isolation`, `label_by`, `packages`, `rollout_fn_ref`, `subagent_datum_sampler`), for a subclass that skips
  the constructor. A group reads the settings that a partial workflow config may lack through
  `platoon.train.workflow.settings.GroupSettings`: a missing one reads as `WorkflowConfig`'s default, which is the
  default each reader had.
- **A trainer on `TrainerBase` keeps its steps atomic and its updates failing closed** (experimental): the defaults of
  `TrainerNeeds.atomic_steps` and `fail_closed_updates` are True, so a trainer of 0.4.0's experimental API that keeps
  `TrainerBase.needs` now saves no `<step>_interrupted` checkpoint of a step that sent training work, and raises
  `UpdateRejected` on a skipped or nonfinite optimizer result of a learner of the run's; its `needs()` may turn either
  off. A step that sends training work by hand, outside `run.learner`, says so (`run.clock.step_open = True`), and
  `run.learner` is built once the run's clock started (`ready()`). `TrainerBase.close` runs after a failure too.
- The rl trainer's step ends through the step hooks' sequence (`platoon.train.trainer.hooks.finish_step`), which runs
  its publication and its clock's advance as before when no hook is registered: `TrainStep.publish` takes the step's
  end and returns who published, and `TrainStep.advance` and `TrainStep.before_first_update` are gone, the barrier of
  the starting policy's evaluation being the learner's (`Learner(before_update=...)`, awaited by `PolicyUpdate.apply`,
  whose step tally is now optional). The rl trainer's `close` stops its recovery pruning after steps that ended
  (`run.ended`), as before, and does nothing after a failure.
- **Logger names follow the modules.** A log line names the module that logs it (the `%(name)s` of platoon's log
  format), so the trainer's lines no longer come from `platoon.train.tinker.rl` or
  `platoon.train.tinker.workflows.group_rollout_workflow` but from, say, `platoon.train.trainer.loops` (a step's line),
  `platoon.train.trainer.rl` (a start's), `platoon.train.workflow.members` (a group's stragglers) or
  `platoon.train.workflow.telemetry` (a rollout worker's). A filter or a level set on `platoon.train.tinker` no longer
  reaches them: set it on `platoon.train`.
- `import platoon.train.tinker` no longer imports the trainer, and with it datasets and transformers: it loads about
  half the modules it did. It still loads tinker, torch and tinker_cookbook: its init applies the rollout worker
  invariants first, from `platoon.train.workers.env`, and the init of `platoon.train.workers` imports the protocol.
  `platoon.train` exports the trainer, its config classes, the batch transforms, `GroupRolloutWorkflow` and
  `current_entry` lazily, and neither it nor `platoon.train.config_defs` loads tinker or torch.
- Rollout workers run `python -m platoon.train.workers.entrypoint`. Their fork server preloads the modules it did
  (the trainer's among them), so a forked worker starts as it did. A worker started by exec no longer imports the
  trainer before its rollout, nor datasets, pandas and requests with it, which a forked worker holds from its server:
  under `worker_python_env_isolation`, a rollout's Python overlay can now replace those in an exec'd worker, not in a
  forked one.

### Fixes

- The `process` sandbox provider's `reap(labels)`, which a run calls before it starts and as it exits, and
  `platoon sandbox reap` with it, returns only once the processes it killed have exited, or after 10 seconds, with a
  warning that names those still running. It returned once it had sent them SIGKILL, while they could still be giving
  back their memory, files and ports.

### Deprecations

- **The module paths under `platoon.train.tinker` that moved.** Each still imports, and each name read through it
  warns once with a `PlatoonDeprecationWarning` that names its new module, until platoon 0.6.0 removes them. A name
  read through an old path is the moved object itself, so `isinstance` checks, pickles and the `module:Class` a config
  names keep working. Importing an old module imports the trainer, as it did before (`platoon.train.tinker` itself no
  longer does: above), so that no module of the trainer is first imported while a patch made through an old path is
  active. A module that was split still has the names it imported, `from <old path> import *` binds what the new
  module exports (a split module's public names that moved), the submodules that an old package's init imported are
  its attributes, and a name set or deleted through an old path, as `monkeypatch` and `mock.patch` do, is set or
  deleted where the moved code reads it, and undoing the patch puts back what was there, also for a `mock.patch` made
  with `create=True` or nested in another patch, by unittest.mock or by the `mock` package. A method of
  `PlatoonTinkerRLTrainer` or `GroupRolloutWorkflow` that moved is a module function that its class keeps under its
  name: patch the class or an instance, as before, not the module function. What stays different through an old
  path: a patch through `platoon.train.tinker.rl` also reaches the moved code of `.permits` and `.censoring`, and one
  through either of them reaches rl's, as they share new modules. Patching an old module's namespace
  (`mock.patch.dict(vars(module))`, `monkeypatch.setitem(vars(module), ...)`) does not reach the moved code: patch the
  attribute instead. An old module's `__all__` is computed when it is read, not held in the module. Where each went:
  - `platoon.train.tinker`'s names (`PlatoonTinkerRLTrainer`, `Watchdog`, the config classes, the batch transforms):
    `platoon.train`;
  - `platoon.train.tinker.rl`: by stage, `PlatoonTinkerRLTrainer` and `Watchdog` to `platoon.train` (they live in
    `platoon.train.trainer.rl` and `.signals`), the loops' exceptions to `platoon.train.trainer.signals`,
    `TrainLoopSharedState` to `platoon.train.trainer.state`, `PlatoonTinkerDataloader` to `platoon.train.dataloader`,
    `compute_training_metrics` to `platoon.train.objectives.policy_gradient`, and the rest to `platoon.train.trainer`,
    `.policy`, `.step`, `.dispatch`, `.checkpoints` and `.evaluation`;
  - `platoon.train.tinker.config_defs`, `.dataset_order`, `.task_records`: `platoon.train.config_defs`,
    `.dataset_order`, `.task_records`; `platoon.train.tinker.train`: `platoon.train.launch`;
  - `platoon.train.tinker.batch_transforms`, `.staleness`, `.censoring`: `platoon.train.step.batch_transforms`,
    `.staleness`, `.metrics`;
  - `platoon.train.tinker.dispatcher`, `.permits`: `platoon.train.dispatch.dispatcher` (the dispatcher's state file,
    `save_state`, `read_state` and `plan_resume` among it, to `platoon.train.checkpoints.dispatch_state`) and
    `platoon.train.dispatch.permits`;
  - `platoon.train.tinker.kept_groups`, `.checkpoint_pruning`, `.evaluation_record`:
    `platoon.train.checkpoints.kept_groups`, `platoon.train.checkpoints.pruning`, `platoon.train.evaluation.record`;
  - `platoon.train.tinker.workflows` and `.workflows.group_rollout_workflow`: `platoon.train.workflow`
    (`GroupRolloutWorkflow`, `RolloutWorkflow`, `current_entry`, `group_baselines`) and its modules, and the group's
    credit to `platoon.train.credit.group` (`group_baselines` lives there);
    `platoon.train.tinker.workflows.base`: `platoon.train.workflow.protocol`; `platoon.train.tinker.start_limit`:
    `platoon.train.workflow.start_limit`;
  - `platoon.train.tinker.workers` and its modules, and `platoon.train.tinker.worker_env`: `platoon.train.workers`,
    its modules, and `platoon.train.workers.env`;
  - `python -m platoon.train.tinker.workers.entrypoint` and `python -m platoon.train.tinker.train` still run, with the
    warning: run `python -m platoon.train.workers.entrypoint` and `python -m platoon.train.launch`.

  After upgrading under a run that is training, restart it, so that its rollout workers run the code it does.

## 0.4.0 (2026-10-25)

### New features

- **Task samplers you choose per environment, and an adaptive one (experimental).** Each environment entry may set
  `task_sampler` and `task_sampler_options`: under `train.task_sampler: weighted` (a mixture's default), the run hands
  each slot to an entry by weight, as before, and the entry's sampler chooses which of its tasks fills it. `uniform`,
  the default, draws the entry's tasks in the seeded passes it always has; `pass_rate` chooses the task whose next
  group is most likely to mix passes and failures, so that more groups train, from the per-task statistics: Thompson
  sampling on each task's pass rate, with a fifth of the draws going to the tasks drawn least recently and at most
  three draws of a task per pool-sized window. `train.task_sampler: pass_rate` chooses over every task of the run
  instead. Each step logs what it drew and why, the informative share it predicted and the one it got, how
  concentrated its draws are, and how long it took to decide (`train/sampler/...`). It is experimental: it becomes the
  recommendation only after a GPU comparison with `uniform` shows more informative groups at an unchanged held-out
  pass@1
  ([Choose tasks by their pass rates](docs/how-to/train-and-eval.md#choose-tasks-by-their-pass-rates-experimental)).
- **Write your own task sampler (experimental).** A sampler is a small class, registered with
  `@task_sampler(name, options=...)` from an entry's `package`, or named as `module:Class`, at the run level or an
  entry's, with typed options (`train.task_sampler_options`, or the entry's `task_sampler_options`) checked against its
  schema before the run starts (core's samplers' already when the config loads) and recorded in `lock.json`. A sampler
  that reads no statistics draws when the order reaches it, as the built-ins always have; one that reads them decides
  when its group can start, among the tasks whose label has room, reproducibly from its seed and its decision count.
  The API and the per-task statistics types it reads are in the new `platoon.experimental` module, and
  `platoon.testing.check_task_sampler` checks a sampler against the contract
  ([Write a task sampler](docs/extending/task-samplers.md)).
- **Train only on the groups that carry signal, with refill (experimental).** `train.group_admission: informative` (or
  an environment's own `group_admission`) decides when each training group settles whether it counts toward the batch: a
  group whose workflow says it trains nothing (its rollouts scored alike, or it carried no datums) is rejected, and one
  whose rollouts scored alike frees its place at once, so that another group starts while the trainer works (one without
  datums, which can end at once, keeps its place until a step takes it). A step trains the first `batch_size` admitted
  groups that are not stale, and stops waiting once it has counted `refill.max_groups_per_step` (3.0) times `batch_size`
  groups, training what it admitted. `group_admission_options: {band: [lo, hi]}` also rejects groups whose pass rate is
  outside the band. `all`, the default, admits every group, as before. A step then trains up to several times the
  informative groups, and only those, at more rollouts a step: up to the cap's multiple of the groups it counts, and
  where the trainer bounds the run, the groups that start while it trains, which the horizon and the rollout permits
  bound, not the cap; each step logs what it admitted and rejected, and why (`train/admission/...`). Under a loss whose
  objective is not zero at zero advantages, a rejected group's advantage-independent terms go too, and loading warns;
  beside a rule, `num_minibatches` must be 1. Custom rules register with `@group_admission` or are named as
  `module:Class`; the API is in `platoon.experimental`, and `platoon.testing.check_group_admission` checks a rule. It is
  experimental: it becomes the recommendation only after a GPU comparison ([Train only on groups that carry
  signal](docs/how-to/train-and-eval.md#train-only-on-groups-that-carry-signal-experimental), [Write a group admission
  rule](docs/extending/group-admission.md)).
- **Rollout permits that come back as each rollout ends, by default.** A training rollout returns its permit
  (`train.max_concurrent_rollouts`) as soon as it ends rather than when its group's workflow returns
  (`train.permit_release: rollout`, the default), so the next group starts on the permits a group's finished rollouts
  give back instead of waiting for its slowest one; on long agentic tasks a quarter to a third of the budget sat idle
  that way. `GroupRolloutWorkflow` runs each member on its group's lease, so a group starts on its first free permit
  and its members as the rest free (rolling starts), each member's straggler deadline shifted by its own start delay; a
  workflow that does not say it runs its rollouts on the lease gets each group's permits whole, as before. Groups are
  drawn, decided, credited, admitted and resumed as before, a partly finished group counts whole toward the horizon,
  and a run at `max_staleness: 0` with the default budget trains the same datums in the same order. More groups are
  open at once, so more long groups can go stale: the run warns when they do, and raising `max_staleness` by one is the
  documented answer. The default changes ahead of a GPU comparison of the two settings; `train.permit_release: group`
  keeps the behavior of earlier releases, event for event (RFC 0020;
  [Rollout permits](docs/how-to/train-and-eval.md#rollout-permits); what changes for a config that sets nothing is
  under Changes, below).
- **Run a workflow's own rollouts on the lease (experimental).** `platoon.experimental` exports `rollout_lease()` and
  `RolloutLease`: a workflow runs each rollout in `async with rollout_lease().rollout(i):`, whose permit comes back
  when the block exits, and declares `runs_rollouts_on_lease = True` for rolling starts. In a training draw a block for
  an index outside the draw's size raises `IndexError`, and a second block for an index `RuntimeError`, under either
  setting; outside a training draw, as in an evaluation, the lease is a no-op ([Run rollouts on the
  lease](docs/extending/rollout-leases.md)).
- **See what keeps groups from starting.** Every training run logs, each step, the share of the step in which the
  dispatcher could not start a group and why (`train/dispatch/blocked_share/{permits,horizon,labels,plan,sampler}`),
  the permits held (`rollout_client/permits_held_mean`), the groups waiting for a worker-startup slot under process
  isolation (`rollout_client/startup_gate_waiting_mean`), and, where a straggler policy applies, what it censors
  (`train/batch/censoring/*`, and each label's: the share of rollouts it cancelled, and over mixed groups the pass
  rate of each group's slowest valid rollout against the others'). It warns, once per start of training, when more
  than 2% of the groups the last 10 steps consumed went stale (so from the tenth step on), naming what held the
  dispatcher back and the fixes, and when more of `GroupRolloutWorkflow`'s rollouts ran at once than
  `max_concurrent_rollouts`.
- **Rollouts start without the work that each used to do again, or wait for.** Two costs fell on a rollout's start.
  A CLI harness's first rollouts on a node waited while one of them copied the harness's toolchain to node-local disk
  (about 36,000 files and 800 MB for OpenHands), at every start and every resume in a new job: `platoon train` now
  stages its toolchains while it opens the training service and saves at launch, and starts no rollout, training's
  or an evaluation's, before that is done (`platoon eval` while it lists its tasks), so that no rollout waits for the
  copy and the first ones wait only for what of it outlasts the launch save; it waits ten minutes at most, and not
  past a shutdown request. And each rollout worker imported, after its fork, what its environment and harness import
  when they open, about 3,700 modules for an in-process ORS environment played by CodeAct, and parsed the
  environment's task file again: the workers' fork server now does that once, before it forks them, from what the
  config names (the environment's module and the openreward SDK, each function harness's and program's module, the
  registry lookups, an in-process environment's task lists for the run's splits), and starts as soon as the training
  client is open. It keeps only what leaves it fork-safe: a module or a build that starts a thread or a process,
  leaves a file, socket or event loop open, or seeds a numpy generator from the OS is left out with a warning that
  says why, and the workers import it themselves. `train.workflow_config.worker_preload` adds modules that an
  environment imports only when it opens. In CPU benchmarks of TextCraft-Synth played by CodeAct with Qwen3-4B's
  renderer, a forked worker reached its agent in 0.03 s instead of 7-9 s, 25-30 s and 48-56 s with 32, 128 and 256
  workers starting at once, with 149 file system calls on the shared filesystem instead of 17,134 and 30 MB of memory
  of its own instead of 290 MB. In CPU emulations of an OpenHands first wave whose toolchain copy took 77-90 s, the
  first sandbox could open 8-17 s sooner after the training client opened, and 46-53 s sooner behind a 38 s launch
  save ([Rollouts are slow to start](docs/how-to/troubleshooting.md#rollouts-are-slow-to-start)).
- **A limit on rollouts starting at once (experimental).** `train.max_starting_rollouts: N` lets at most `N` rollouts
  set up their workers, sandboxes and harnesses at once, training's and evaluation's together, first come, first
  served: a `GroupRolloutWorkflow` rollout takes a start slot once it holds its permit and gives it back as its first
  model call is issued, as it ends, or after `worker_startup_budget_seconds` + 300 s without a call. It is for
  start-ups that contend for a shared capacity, such as sandboxes that each extract a root filesystem; off by default
  (`null`), it changes nothing. It shortens no one-time start-up work (the fork server's import, a harness toolchain's
  copy to each node, the modules each worker imports), and with any value a run's whole first wave reaches the model
  later. Each step logs the rollouts starting and waiting to start, their waits and why slots came back
  (`rollout_client/start_limit/...`), and the run warns when slots come back on their timer, or mostly only at their
  rollouts' end ([Rollout start-ups](docs/how-to/train-and-eval.md#rollout-start-ups-experimental)).

### Changes

- `train.task_sampler` takes registered names and `module:Class`. An unknown name is still refused when the config
  loads, in the same report as the file's other problems, with a longer message that begins as before, unless an
  entry names a `package` that could register it; it is then refused before the run starts.
- A full checkpoint's `tasks/dispatch-*.json` is format 2: it adds the row each outstanding draw drew, the count and
  state of each task sampler, the run's and each environment's under `weighted` (and `auto` for a mixture), and the
  slots waiting for an environment's adaptive sampler. A resume reads platoon 0.3.0's format 1 as before, and goes on
  exactly where its run stood; platoon 0.3.0 cannot resume exactly from a checkpoint that 0.4.0 wrote (it draws from
  the checkpoint's batch, as from a checkpoint of 0.2)
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- The per-task statistics record which task sampler drew each group, and why: `GroupRecord.sampler` names the
  sampler that chose the task (the run's sampler as the config names it, or an entry's as `<sampler>@<entry>`, such as
  `uniform@<entry>` for an environment's `uniform` under `weighted`), with the draw's `reason` and `score`, and so do
  the draw events of `tasks/events.jsonl`, every one of them.
- Every training run's `lock.json` records its task samplers, the run's and each entry's it asks, as components of
  `kind: task_sampler` with their options, defaults included; and every run logs what its samplers drew since the last
  step, by reason, and how long their decisions took (`train/sampler/draws`, `train/sampler/draws/<reason>`,
  `train/sampler/decision_ms`, and an entry's under `train/sampler/environment/<entry>/`).
- The per-task statistics take in settled groups in the order they were drawn, at each step's metrics, each full
  checkpoint and each adaptive decision, not in the order they completed, so their values no longer depend on
  completion order within a step.
- The per-task statistics have a sixth fate, `rejected`, a group admission rule's: `platoon tasks show` prints it
  after `lost`, and `platoon tasks export` has a `fate_rejected` column after `fate_lost`, so the five columns after it
  (`group_size` to `newest_version`) move one place right. A rejection's `fate` event in `tasks/events.jsonl` names the
  rule and its reason (`rule`, `admission`).
- `GroupRolloutWorkflow` declares `gives_verdicts = True`; a workflow class of your own that gives each group a
  verdict declares it too, so that `group_admission: informative` loads beside it.
- A full checkpoint of a run under a group admission rule records in its row of `checkpoints.jsonl` where the order
  stood (`next_position`): a resume that cannot read the checkpoint's state of the draws draws on from there, since
  such a run's steps draw more positions than their batch.
- `GroupRolloutWorkflow` declares `runs_rollouts_on_lease = True` and runs each member in a block of its group's
  lease; a subclass that overrides `arun_episode`, `_run_group_rollouts` or `_run_group_member`, itself or through a
  mixin, reads as `False` unless its own class body sets it. Under `permit_release: group` the blocks hold nothing,
  but a training group of a subclass that runs a member index twice, as a retry of the whole group in
  `arun_episode` does, now fails with `RuntimeError` under either setting: a retry stays inside the member's block.
  A member's rollout sees no lease, so a group that it runs in turn takes no permits of its own and runs as before.
- **Breaking: rollout permits come back as each rollout ends, unless `train.permit_release: group`.** The default of
  `train.permit_release` is `rollout`, so `train.max_concurrent_rollouts` counts the rollouts running, no longer the
  whole groups it holds. Where the permits bound a run, more groups run at once and start sooner, steps get shorter, and
  more long groups can be dropped as stale: when more than 2% of the groups the last 10 steps consumed went stale, the
  run warns and advises `max_staleness + 1`. There its rollouts are requested in another order, some on a newer policy,
  and a restart redoes more of them. In process, a rollout's permit also waits, at most 60 s, for the sandbox and env
  session teardowns that platoon finishes in the background, so that no more sandboxes are alive at once than the budget
  (under `group` such a teardown can still outlive its group, as before). A run whose permits never bind, as at
  `max_staleness: 0` with the default budget, trains as before. Each start of training under `rollout` says so in a line
  of its log (`Each rollout returns its permit as it ends (train.permit_release: rollout) ...`). `train.permit_release:
  group` restores the behavior of earlier releases ([The previous
  behavior](docs/how-to/train-and-eval.md#the-previous-behavior)).
- Under `train.permit_release: rollout`, `rollout_client/rollouts_in_flight_configured` is `max_concurrent_rollouts`
  itself, so `concurrency_achieved_fraction` reads as the permits' busy share, and `groups_in_flight` can exceed the
  budget over the group size. The straggler settings' documentation calls them a deadline against runaway rollouts,
  not a throughput lever.
- Under `train.permit_release: rollout`, an in-process rollout's permit comes back once the sandbox and env session
  teardowns that platoon finishes in the background have ended too, such as one that a straggler cut interrupts or an
  ORS environment's that waits for its code to end, so that no more sandboxes are alive at once than
  `max_concurrent_rollouts`. The permit waits for them at most 60 s, and the run warns of a teardown that runs longer;
  a cancellation ends the wait and no teardown, and a rollout that had ended keeps its outcome. Under `group` nothing
  waits for them, as before ([Rollout permits](docs/how-to/train-and-eval.md#rollout-permits)).
- A staged harness toolchain has a stamp beside it, `.<copy>.staged`, that records the tree hash it was checked
  against and the jobs that checked it. Where the copy outlives its job (`PLATOON_TOOLCHAIN_CACHE` on node-local disk
  that persists), the first staging of each later job hashes the copy again and keeps it when it matches its
  manifest; before, a kept copy was used unchecked. One that no longer matches is copied again beside it, and the old
  copy is set aside whole (`.<copy>.<hex>.stale`), never removed, since rollouts of another job may still run from it.
  A copy staged by an earlier platoon is checked once. A cache that platoon cannot write to (read-only, another
  user's, a full disk) records no check, so its kept copy is used unchecked, as before, with a warning.
  `platoon.harness.toolchains.prestage` stages toolchains ahead of rollouts.
- The rollout workers' fork server takes 4-8 s longer to start and holds 150-230 MB more memory, which the workers it
  forks share (14-20 s against 10-13 s, and 890-930 MB against 670-740 MB, in CPU benchmarks of TextCraft-Synth played
  by CodeAct with Qwen3-4B's renderer), once per run and while the trainer saves at launch; each worker holds far less
  of its own.
  Its first log line counts the modules it imported for the environments beyond the worker's stack (`imported the
  worker's stack and 9 module(s) the environments open with`). An environment variable that one of those imports
  sets is set in each worker too. Under `worker_start_method: exec`, `rollout_isolation: inprocess` and
  `worker_python_env_isolation` nothing is imported ahead, as before, and a `worker_preload` set there is warned
  about.

### Fixes

- A `GroupRolloutWorkflow` group without a straggler policy no longer leaves its other rollouts running when one of
  them raises (as a subclass's own `arun_episode_single` can): it cancels them and raises once they have unwound. They
  ran on after the group had failed and its rollout permits had come back, so more rollouts could run at once than
  `train.max_concurrent_rollouts`, on work that no step trains. A cancelled group likewise returns only once all its
  rollouts have unwound, not once the first has.
- A resume under `train.task_sampler: uniform` draws again exactly the row it had drawn; it could take the listing's
  form of the row, whose nested values differ in their empty keys.
- A rollout worker that is cancelled, as by a straggler cut or a stop, no longer leaves its readiness and diagnostics
  files in node-local scratch, where they stayed until a later run found them a day old.
- **A model service that sends no routing ids is warned about once, not for each rollout.** With
  `rollout_isolation: process`, the trainer warned about every rollout whose model calls did not all get a routing
  id, which a server with rollout routing replay (R3) sends with each sampled sequence. With a service that sends
  none, as a dense model's does, the trainer logged a warning for each rollout: 18,125 in one 70-step run. Now it
  warns once for each start of training, about training and about evaluation rollouts each, that the service sends no
  routing ids: expected for a dense model, but no rollout routing replay for a mixture-of-experts model. It still
  warns about each rollout that got routing ids for some of its model calls but not all, or for none after the service
  has sent some, and it reports the first rollout to get any after rollouts without them, naming the first few of
  those. The metrics `train/rollout_worker_routing_id_coverage` and `eval/rollout_worker_routing_id_coverage` are
  logged as before.

### Removed

- **Breaking: `train.task_order` is gone.** Platoon 0.3.0 renamed it `train.task_sampler` and deprecated the old name
  for removal in 0.4.0. A config or an override that still sets it fails as a configuration error (exit 78), with
  `train.task_order: unknown key; platoon 0.4.0 removed it; use train.task_sampler`; so does a config copied from
  platoon 0.2.0's `config show --defaults`, which spells `task_order: auto`, even beside a `task_sampler`, where 0.3.0
  ignored it. Set `train.task_sampler` instead: it takes the old name's values, `auto` and `weighted`, which draw as
  they did ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).

### Install and upgrade

```bash
pip install -U 'platoon[tinker,ors]>=0.4,<0.5'
# with the TextCraft extension, name it too:
pip install -U 'platoon[tinker,ors]>=0.4,<0.5' 'platoon-textcraft[platoon]'
```

- **To upgrade a project that pins one minor release**, move its range to 0.4, such as `platoon[tinker,ors]>=0.4,<0.5`.
  Platoon 0.4 refuses the components of an installed extension whose requirement on platoon excludes it, such as
  `platoon>=0.3,<0.4`, with an error that names both versions, until a release of the extension admits 0.4;
  meanwhile, `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` loads them anyway
  ([Compatibility and provenance](docs/extending/index.md#compatibility-and-provenance)).
- **A config that sets `train.task_order` no longer loads:** use `train.task_sampler`, which takes its values
  (`platoon config validate` names the key).
- **`train.permit_release` now defaults to `rollout`.** Where the permits bound a run, more groups run at once and
  steps get shorter, and more long groups can be dropped as stale: if the run warns, raise `train.max_staleness` by
  one. `train.permit_release: group` trains as 0.3 did
  ([The previous behavior](docs/how-to/train-and-eval.md#the-previous-behavior)).
- **The TextCraft extension's release for platoon 0.4 is `platoon-textcraft` 0.3.0**, whose `platoon` extra requires
  `platoon[ors]>=0.4,<0.5`. Its code and data are those of 0.2.0. The release for platoon 0.3 stays 0.2.0: there, keep
  the extension within 0.2 (`platoon-textcraft>=0.2,<0.3`), since platoon 0.3 refuses the components of 0.3.0, also
  when it is installed without its extra. An upgrade of platoon alone leaves 0.2.0 installed, which pip does not flag
  and platoon 0.4 refuses: upgrade the two together, as above
  ([The TextCraft extension](docs/getting-started/install.md#the-textcraft-extension)).

## 0.3.0 (2026-10-20)

Each environment of a training mixture can set the training settings of its own groups, training keeps per-task
statistics and logs pass@k, a dispatcher starts each training group when it can start, and a resume takes up exactly
where the run stood. A label at its `rollout_group_limits` no longer paces the run, which changes what such a config
trains. Harbor verifiers that ask for root can run as root (`verifier_user: honor`), and the `openhands` harness can
condense at a token count. `train.task_order` is now `train.task_sampler`, with a new `uniform` sampler. The forms that
platoon 0.2.0 deprecated are gone, but `strategy` stays, as the unit of `every`. The `hpc` extra takes hpc-sandbox 0.4,
and the TextCraft extension's release for platoon 0.3 is `platoon-textcraft` 0.2.0.

### New features

- **Each environment of a mixture sets the training settings of its own groups.** An entry's `workflow_config` is laid
  over `train.workflow_config` for its groups: its `group_size`, its baseline (`leave_one_out_baseline`), its
  `min_successful_group_size`, its straggler defaults, its filters (`filter_errors`, `filter_message_actions`) and
  subagent datum sampling, and its rollout settings (`rollout_config`: `max_steps`, `timeout`, sampling and the rest),
  so a dataset's calibrated step budget and a slow environment's fewer, longer rollouts sit beside the run's. Its
  rollout settings are its evaluations' defaults, below `eval.workflow_config` (and in `platoon eval`, below
  `inference.workflow`), and `eval_overrides.workflow_config` sets what its evaluation rollouts change (every split but
  `train`). An entry's `max_staleness` raises the staleness bound of its own groups, while `train.max_staleness` stays
  the bound on how far rollouts run ahead, and recovery pruning keeps the checkpoints within the largest bound. Loading
  refuses a key that acts on microbatches or on the run's rollout workers with where to set it, an entry bound below the
  run's, and an entry's settings beside a workflow class that does not apply them (exit 78). Named environments may ship
  rollout settings (`workflow_config.rollout_config`) only. `platoon config show` prints each entry's effective settings
  with the layer each came from (`--entry NAME` for one), `platoon config validate` and `platoon train` say what each
  entry changes, and `lock.json` records it (`entry_settings`). `max_concurrent_rollouts` counts rollouts that groups of
  different sizes share, and each step logs each label's `stale_groups` and `num_steps`
  ([Settings per environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **Training keeps per-task statistics.** Each settled group of a training task is recorded with the workflow's
  verdict on what it trains (`trained`, `uniform`, or why it trains nothing) and what each rollout returned, passes
  judged as `platoon eval` judges success: on the reward the environment scored, before any reward processor, against
  the entry's new `success_threshold` (1.0 by default; a named environment may ship its own), which only these
  statistics read (`platoon eval` keeps `inference.workflow.success_threshold`). Each step logs each label's mean
  expected pass@1 and pass@G under decayed per-task counts (`train/tasks/<column>/<label>/pass_at_1`, `pass_at_G`), the
  share of its tasks seen, its tasks by pass-rate band and those without evidence, and the kinds of the groups it
  trained on (`train/batch/informative_groups`, `all_pass_groups`, `all_fail_groups`, `uniform_mixed_groups`, also by
  label). Each full checkpoint saves the statistics to `tasks/` in the run directory and names the file in its
  `checkpoints.jsonl` row, a resume loads them before its launch save (from a checkpoint without them, it goes on from
  the newest older one that has them, else from empty, and `lock.json` lists the start under `tasks_state_lost`),
  recovery pruning deletes a pruned checkpoint's file, and `tasks/events.jsonl` logs every draw, settle and fate. The
  statistics never fail training. `platoon tasks show` and `platoon tasks export` read them
  ([Per-task statistics](docs/how-to/train-and-eval.md#per-task-statistics)).
- **A resume takes up exactly where the run stood.** Each full checkpoint saves where the run's draws stand in a file
  of its own (`tasks/dispatch-s<start>-<checkpoint>.json`, which its row names as `dispatch`), and the groups that
  settled and that no step had trained yet (`tasks/kept-s<start>-<checkpoint>.json.gz`): a resume trains those as they
  are, a failed one as failed, without rolling them out again, draws again first the tasks whose groups were running
  or waiting, and goes on with the order where it stopped. So every position of the order is consumed exactly once,
  trained, dropped as stale, failed or skipped, and an interrupt or a crash no longer reruns hours of a slow
  environment's groups or loses the groups in flight, also when the per-task statistics stopped or could not be
  written. Every full save writes the groups it keeps again, about ten bytes a token, in a thread beside training;
  `checkpoint.keep_recovery` deletes them with their checkpoint. A resume whose sampling inputs changed (the task
  sampler, the batch size, the label column, the entries' order, an entry's weight, seed or tasks) warns, records it
  in `lock.json` (`sampling_changed`) and goes on; one from a checkpoint of an earlier platoon draws from its batch,
  as before ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- **`train.task_sampler: uniform`** (`train.task_sampler` is the new name of `train.task_order`, below) draws every
  training task of every entry in seeded shuffled passes, with no entry slots: every task equally often, as
  `weight: tasks` does, with batches as the shuffle mixes them rather than balanced by entry. It reads no entry's weight
  or seed: a config that gives an entry a numeric weight does not load (exit 78). For a single environment it draws the
  batches `auto` draws, in epochs that hold every task
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Harbor verifiers that run as root.** Some Harbor tasks grade as root (`[verifier] user = "root"`) and run the
  agent's code as another user, nobody as a rule, so that it cannot read the files the grader keeps for itself, such
  as held-out answers. In a rootless sandbox their tests run as your own uid, which cannot switch users: they score
  every attempt 0, the reference solution's included, or run the agent's code where it can read those files.
  `verifier_user: honor` now runs such a task's separate verifier as root, in a sandbox started with
  `verifier_provider_options` laid over `provider_options`: on Singularity, `{rootfs_layout: copy, fakeroot: true}`,
  whose `--fakeroot` maps root to your uid and your subordinate ids to the other users, so nothing gains a privilege
  on the host. On Singularity it needs hpc-sandbox 0.4, whose provider takes `fakeroot` and gives back what the other
  users write before it deletes a sandbox, and which the `hpc` extra now installs (below); with an older hpc-sandbox,
  such as 0.3.0, such a run is refused.
  `verifier_user` (`refuse`, `ignore` or `honor`) decides for `[verifier].user` when it is set, and without it
  `agent_user` decides for both users, as before. With a fetched dataset, set `refuse: {agent_user: false}` too: the
  fetch otherwise leaves out every task that names a user, and a run with `honor` warns of the root verifiers it left
  out. A run refuses, when it lists its tasks (exit 78), a root verifier whose sandboxes cannot give it root and the
  other users (`singularity` without `fakeroot`, or with an hpc-sandbox that does not take it, `enroot`, which maps only
  your uid, and `process`), and, for a fetched dataset, a `[verifier].user` that `honor` cannot honor (another user
  than root, or a root verifier of a task graded in its agents' own sandbox), which a dataset directory skips with a
  warning. Every started root verifier's sandbox is checked to run as root with every user up to nobody mapped, and
  to let nobody run the image's shell, else the grade is an `error` (`verifier_start`). A root verifier grades as
  Docker would: with umask 022, Docker's modes for the directories platoon makes and for a working directory the
  image lacks, the image's own modes for those it has, its artifacts readable by every user, and the outcome records
  `verifier_user: root`. Separate verifiers' sandboxes are reaped with their own options too
  ([Verifiers that run as root](docs/how-to/sandboxes-and-harbor.md#verifiers-that-run-as-root)).
- **The `openhands` harness condenses at a token count.** Its new option `condenser_max_tokens` gives its summarizing
  condenser a token trigger, and runs that condenser whatever `condenser` says: before each model call, it condenses
  once the prompt the agent would send is longer than that many tokens, summarizing the oldest turns after the system
  prompt and the task until the rest fits in half the trigger. `condenser_max_tokens: 104857` condenses at 80% of a
  131072-token context window; a trigger that a prompt passes with nothing left to forget, such as one below the
  agent's first prompt, can never be met, and is set aside with a line in the harness log. The runner asks the
  gateway for each count (`POST /v1/messages/count_tokens`), so it is the length of the prompt the gateway renders
  with the policy's renderer; where a hosted model's provider gives no count, the SDK's estimate counts, and a prompt
  that still overflows the window condenses as before. The summaries are the agent's `aux` calls (below), trained
  with the rollout's advantage. The default, 0, sets no trigger and counts no tokens
  ([`openhands`](docs/tutorials/builtin-harnesses.md#openhands)).

### Changes

- **A dispatcher starts each training group when it can start.** The producer that drew a batch's tasks ahead of the
  rollout workers, the pending queue and the pool of workers are replaced by a dispatcher that draws a task only when
  its group can start: within the horizon of `train.max_staleness` (at most `max_staleness + 1` batches of groups
  started and not yet trained), with rollout permits for the whole group, and with room in its label's
  `rollout_group_limits`. A run without `rollout_group_limits` trains the groups it trained before; at
  `max_staleness: 0` each step trains exactly the groups of its batch, unless a label's limit holds one of them back.
  The run starts no group past its last step ([Settings per
  environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **Breaking: a label at its `rollout_group_limits` no longer stalls the others, and its groups start as its limit
  allows, not as the steps do.** In every config that sets limits, a draw whose label is at its limit waits in the
  label's backlog, outside the horizon: it neither takes a place in a batch nor counts toward the batches
  `train.max_staleness` lets the run draw ahead. Past four batches of them, the label's new draws are skipped and
  counted, and the run draws only while some label has room. While a capped label has draws waiting, each of its
  running groups holds a slot of the horizon for the next one, which starts as soon as the group ends, as far as the
  capped labels' groups and those slots fit in `max_staleness` batches; at `max_staleness: 0` there is no slot to
  hold, and a capped label starts at most its limit of groups a step. So a capped label whose limit cannot keep up
  with its share trains less than its weight, and its draws still waiting when the run ends never start;
  `train/batch/parked_rows` counts the draws that wait, each step, and `train/batch/skipped_rows` those skipped.
  `platoon train` and `platoon config validate` warn of a `max_concurrent_rollouts` that the capped labels' groups can
  fill. There is no switch: in 0.2, a label at its limit held back the other labels' rows, and so paced the whole
  run. This changes what a config with `rollout_group_limits` trains: its capped labels' draws that wait are drawn
  past at every bound, `max_staleness: 0` included
  ([Settings per environment](docs/how-to/train-and-eval.md#settings-per-environment)).
- **The `openhands` harness's condenser keeps the task and shows its summary call what happened.** The spec is now
  `openhands@1.54.0.post1`, from the same toolchain, so no site builds it again, and a config that names
  `openhands@1.54.0` gets it; with the default options (no condenser and no token trigger), the agent runs as before.
  With `condenser: llm_summarizing` or a token trigger, the summary call reads each forgotten turn whole: the commands
  it ran, their whole output and an earlier summary entire, where the SDK's condenser showed it 500 characters of each
  and no tool call's arguments, so a summary call's prompt is now as long as the turns it forgets. A condensation that
  resets the whole context keeps the system prompt and the task, which the SDK's summarized away with the rest, and
  one for the token trigger forgets only what it takes to get under half the trigger. A summary call answered without
  text (a tool call, or a completion cut short in its reasoning) still condenses, with a notice in the summary's place
  that the earlier turns were dropped, where the SDK's condenser dropped them and said nothing. A context reset whose
  summary call fails, such as one the gateway refuses for overflowing the context window, is made again with the
  turns cut shorter each time, and if every try fails, the turns are dropped with the same notice, where the SDK's
  condenser stopped the agent. No option brings back the SDK's condenser.
- **The `openhands` condenser's summary calls are `aux` calls.** The runner marks each summary call with
  `X-Platoon-Call-Kind: aux`, and the agent's own turns stay `policy` calls. A summary's trajectory step now says
  `kind: aux` in its `action_misc`, and an evaluation report counts the summaries under `usage.aux` rather than
  `usage.policy`. They train as before, with the rollout's advantage, each condensation still takes one of the
  agent's steps (`max_steps`), and the step counts in training and evaluation metrics still count each summary as a
  step. In a task sandbox, the gateway takes them under the rule below.
- **A harness in a task sandbox may make `aux` calls.** In 0.2 the gateway refused every call but a policy call from a
  harness in a task sandbox, where code the policy runs can read the agent's key. It now takes its `aux` calls, unless
  a gateway step budget (`[budget] enforce = "gateway"`) counts its calls: that budget does not count `aux` calls, so
  under it they are still refused (403). Without one the gateway counts none of the harness's calls, and it samples
  the harness's `aux` calls as its policy calls, also under `sampling_precedence: config`, where an `aux` call
  elsewhere keeps its own parameters; so an `aux` call gets that code nothing that a policy call would not. Its
  `judge` calls are still refused in every case
  ([how a run ends](docs/how-to/sandboxes-and-harbor.md#3-how-a-run-ends)).

### Fixes

- **The `oci` image builder keeps the permissions an image gives other users.** Run by anyone but root,
  `unsquashfs` gave the files it unpacked the image's modes through the host's umask, so under a umask such as 027
  every file of an image the `oci` builder made lost the read and execute bits the image gives other users. Rootless
  sandboxes never noticed, since everything there runs as the files' owner, but a verifier that runs as root and
  switches users could not even run the image's shell. The builder now unpacks under umask 0, in a build directory
  only you can enter, and records `builder_revision` in its receipts. Images it built before are not built again by
  themselves, since their build keys do not change: with `verifier_user: honor`, `platoon train`, `eval`, `rollout`
  and `prepare` refuse the root verifiers' images among them, naming the build that makes them again
  (`platoon harbor images build CONFIG --key KEY`)
  ([Verifiers that run as root](docs/how-to/sandboxes-and-harbor.md#verifiers-that-run-as-root)).

### Removed

- **Breaking: the forms that platoon 0.2.0 deprecated are gone.** A config, an override or a command that still uses
  one fails as a configuration error (exit 78), with a message that names what replaces it and says that platoon 0.3.0
  removed it, such as `train.rollout_group_key: unknown key; platoon 0.3.0 removed it; use train.label_by`
  ([Write and override configs](docs/how-to/configs.md#4-check-the-result-before-you-run-it)). Each form, with what
  replaces it:
  - `rollout_config.propogate_root_success`, the misspelling: `propagate_root_success`;
  - `train.rollout_group_key`: `train.label_by`;
  - `workflow_config.straggler_policy_key` of `train` and `eval`: `train.label_by`, whose labels straggler policies
    key on;
  - `num_concurrent_rollout_workflow_workers` of `train` and `eval`, which counted rollout groups:
    `max_concurrent_rollouts`, which counts rollouts (the old value times the section's `workflow_config.group_size`);
  - `strategy: none` of `eval` and `checkpoint`: `every: 0` (`strategy: step` and `epoch` stay, below);
  - a Harbor entry's `cache`, `image_map` and `toolchain_root`, and the variables `PLATOON_HARBOR_CACHE` and
    `PLATOON_TOOLCHAINS`: the site profile's `cache`, whose `harbor` directory holds the Harbor cache with each fetched
    dataset's image map (`images/<provider>.json`), and whose `toolchains` directory holds the harness toolchains.
    A command that loads a config, or takes the cache from the site, refuses to run while one of the variables is
    set. An entry that serves a dataset directory has no image map: its tasks run their `docker_image` or
    `default_image`, and naming the directory as a `dataset` to fetch has `platoon prepare` build their images;
  - fetch config files of `platoon harbor`, and `platoon harbor fetch`'s `--lock` and `--cache`, which applied to them
    alone: a Harbor entry of a training or evaluation config, and `--workers` for their `workers`;
  - `python -m platoon.train.tinker.restart_wrapper` and `run_with_restart`, whose import raises an `ImportError`:
    `watchdog.max_restarts` in the training config, with which `platoon train` restarts training after any failure.

### Deprecations

- **`train.task_order` is now `train.task_sampler`,** which names how a run draws its training tasks and takes the same
  values: `auto`, the default, draws a mixture in its weighted fair order and a single environment in the dataloader's
  shuffled passes, and `weighted` draws a single environment in the weighted order too. The order a run draws is
  unchanged. `train.task_order` still loads, with a deprecation warning, until platoon 0.4.0 removes it; `lock.json`
  records it and `platoon config validate --strict` fails on it. Beside another `task_sampler`, `task_order: auto` (its
  default in 0.2.0, as a dump of that release's `config show --defaults` spells it) is ignored, with a warning that says
  so, and `task_order: weighted` does not load
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **`strategy` is no longer deprecated.** Platoon 0.2.0 deprecated the `strategy` of `eval` and `checkpoint` for removal
  in 0.3.0; it stays instead, as the unit of `every`, and loads without a deprecation warning. `every` counts in its
  units: `step`, what a section that leaves it out counts, or `epoch`, a pass over the training tasks, so
  `eval: {strategy: epoch, every: 1}` evaluates at the end of each epoch. A config that sets it keeps the meaning it has
  had: a section that sets `strategy` and leaves `every` out triggers its event every 1 of its units, and one that sets
  neither never does. `strategy: none`, which `every: 0` duplicates, is gone (above)
  ([Cadences in steps or epochs](docs/how-to/train-and-eval.md#cadences-in-steps-or-epochs)).

### Install and upgrade

```bash
pip install -U 'platoon[tinker,ors]>=0.3,<0.4'
# with the hpc extra and the TextCraft extension, name them too:
pip install -U 'platoon[tinker,ors,hpc]>=0.3,<0.4' 'platoon-textcraft[platoon]'
```

- **Name every extra and extension you installed.** pip upgrades only what the command names and what that needs, and
  it does not record the extras a package was installed with, so `pip check` passes either way: an upgrade without
  `hpc` keeps hpc-sandbox 0.3, and one without `platoon-textcraft` keeps its 0.1.0 (both below).
- **Replace what 0.3.0 removed** (above): `platoon config validate` reports every removed key or value that a config
  file still sets, each with what replaces it, and refuses to run while a removed variable is set, naming what replaces
  it ([Check the result before you run it](docs/how-to/configs.md#4-check-the-result-before-you-run-it)). A script that
  runs `python -m platoon.train.tinker.restart_wrapper`, `platoon harbor` on a fetch config file, or
  `platoon harbor fetch` with `--lock` or `--cache` stops with exit 78 too, naming what replaces it.
- **Check what trains differently with the same config** (Changes, above): a config with `rollout_group_limits` no
  longer lets a capped label pace the run, so a capped label whose limit cannot keep up trains less than its weight, and
  its new draws are skipped once four batches of them wait; an `openhands` harness with `condenser: llm_summarizing` or
  a token trigger shows its summary call the forgotten turns whole, and an evaluation report counts the summaries under
  `usage.aux`; and a harness in a task sandbox gets its `aux` calls answered and trained, where 0.2 refused them, unless
  a gateway step budget counts its calls. None of these has a switch back.
- **To upgrade a project that pins one minor release**, move its range to 0.3, such as `platoon[tinker,ors]>=0.3,<0.4`.
  Platoon 0.3 refuses the components of an installed extension whose requirement on platoon excludes it, such as
  `platoon>=0.2,<0.3`, with an error that names both versions, until a release of the extension admits 0.3;
  meanwhile, `PLATOON_IGNORE_EXTENSION_REQUIREMENTS=1` loads them anyway
  ([Compatibility and provenance](docs/extending/index.md#compatibility-and-provenance)).
- **The `hpc` extra takes hpc-sandbox 0.4** (`hpc-sandbox>=0.4.0,<0.5`), whose singularity provider has the
  `fakeroot` option that verifiers that run as root need (above). An upgrade that names the extra upgrades an
  hpc-sandbox 0.3 it finds. One that does not keeps it, and platoon 0.3 then refuses `fakeroot`, and with it the root
  verifiers on Singularity, with a message that names hpc-sandbox 0.4. Sandboxes started without `fakeroot` (or
  `--fakeroot` in `extra_start_args`) behave as with hpc-sandbox 0.3.
- **The TextCraft extension's release for platoon 0.3 is `platoon-textcraft` 0.2.0**, whose `platoon` extra requires
  `platoon[ors]>=0.3,<0.4`. Its code and data are those of 0.1.0, but for the dataset card that its Harbor export
  writes, whose example config for platoon no longer sets `image_map`, which platoon 0.3.0 removed. The release for
  platoon 0.2 stays 0.1.0: there, keep the extension within 0.1 (`platoon-textcraft>=0.1,<0.2`), since platoon 0.2
  refuses the components of 0.2.0, also when it is installed without its extra. An upgrade of platoon alone leaves
  0.1.0 installed, which pip does not flag and platoon 0.3 refuses: upgrade the two together, as above
  ([The TextCraft extension](docs/getting-started/install.md#the-textcraft-extension)).
- **Every training run keeps per-task statistics and where its draws stand, with no switch** (above). Its run
  directory gains `tasks/`: with each full checkpoint, a snapshot of the statistics, which its row of
  `checkpoints.jsonl` names (`tasks`), the state of its draws (`dispatch`) and the groups it keeps, about ten bytes a
  token (with `max_staleness: 3`, batches of 32 groups of 8 and rollouts of 16,000 tokens, a save can keep some 130 MB);
  and `tasks/events.jsonl`, a line for each draw, settle and fate, which grows with the run. `checkpoint.keep_recovery`
  deletes a pruned recovery checkpoint's files with it; those of permanent checkpoints stay, as the checkpoints do, and
  without it so do all. Each step logs more series for each label. A run that platoon 0.2 started resumes from its
  checkpoint's batch, as before, with empty statistics and a warning, and its `lock.json` lists the start under
  `tasks_state_lost`.
- **Code of your own.** The rows a workflow gets carry two more keys, `_platoon_seq` and `_platoon_position`, which
  number the training rows the trainer draws, and a loaded `TrainConfig` names its sampler in `task_sampler` alone,
  with `task_order` None. Inside a group, a `GroupRolloutWorkflow` subclass's `self.config` holds the settings of the
  group's entry, which `platoon.train.tinker.workflows.group_rollout_workflow.current_entry()` names; a workflow class
  that is no `GroupRolloutWorkflow` must apply an entry's settings itself and declare `applies_entry_settings = True`,
  or a config whose entries set any does not load into `platoon train` (exit 78). A workflow class of your own that
  returns a plain list of datums gives the per-task statistics no evidence, and one that returns neither that nor a
  `GroupRolloutWorkflow`'s output has its task rolled out again on a resume, rather than its group kept. A sandbox
  provider may say why its sandboxes cannot give a root verifier root and other users, with a `root_refusal(options)`
  class method.

### Documentation

[RFC 0019](docs/rfcs/0019-training-mixtures.md), a draft, designs training mixtures' per-environment settings, per-task
statistics and task samplers. This release ships its first milestones: `train.task_sampler`, settings per environment,
per-task statistics, and the dispatcher with exact resume and the run-level `uniform` sampler.

## 0.2.1 (2026-10-19)

### Fixes

- **A busy node no longer makes the trainer kill rollout workers that were already forked.** On a node whose CPUs are
  all busy, the trainer's event loop could be held up for longer than the fork server's one-minute deadline. When it
  ran again, it gave up on workers the fork server had forked meanwhile, killed them and started them again by exec,
  each importing the worker's stack itself, which loaded the node further. The trainer now reads what the fork server
  sent before it judges a deadline, so an answer, or the hello of a server that finished importing, is used however
  late it is read, and a fork fails only once the server has answered no request for a minute (one that answers the
  requests before it is working through them, in order). A fork server still importing the stack after the worker's
  startup budget is no longer killed and replaced by another that imports it from the start while the workers wait for
  that one: they start by exec meanwhile, and the server forks the next ones once it has imported the stack. If it
  then finds that it cannot serve (a module did not import, a thread runs in it), the trainer logs why and starts no
  other, as for a server that says so in time.

## 0.2.0 (2026-10-19)

The first release on PyPI, numbered 0.2.0 because the original platoon gave its version as 0.1.0, so that the
two never share one. It replaces the original platoon, whose AReaL-based trainer, standalone inference
scripts and plugin layout are gone. The original code is at the tag
[`archive/v0.1`](https://github.com/ApGa/platoon/tree/archive/v0.1), and the code that the
[Recursive Agent Optimization (RAO)](https://arxiv.org/abs/2605.06639) paper's experiments ran on stays on the
[`apga/rao-snapshot`](https://github.com/ApGa/platoon/tree/apga/rao-snapshot) branch.

### What it does

- **Trains agents with reinforcement learning through a Tinker-compatible training service**, hosted Tinker or a
  self-hosted server such as [miles-tinker](https://github.com/neulab/miles-tinker). `platoon train` runs groups
  of episodes per task, builds training data from the exact sampled tokens and centers rewards within each group
  (mean or leave-one-out baselines, with depth weighting for recursive agents). Each step logs its reward and its
  rollouts' outcome counts, and a rollout whose environment could not open its session counts as an error.
  `train.label_by` names the dataset column that labels the tasks, by default the mixture entry: each step's metrics
  and each evaluation's break down by its labels (the 64 most frequent), `train.rollout_group_limits` and the
  straggler policies of training and evaluation key on them, and a column that no training task has stops the run
  before it starts ([Label tasks by a column](docs/how-to/train-and-eval.md#label-tasks-by-a-column)).
- **Any harness.** A harness is a Python function (`@pt.harness`) or a command-line agent described by a TOML
  spec. The built-ins are `platoon-codeact`, `platoon-react`, `platoon-bash`, `mini-swe-agent`, `openhands` (an
  OpenHands SDK agent), and `platoon-oracle` and `platoon-noop`, which run a task's reference solution or nothing,
  to check a dataset before training.
- **One harness per agent, configured by options.** A TOML spec declares its options in an `[options]` table (type,
  default, description, allowed values or range), which `platoon harness show` lists and its launch renders; a
  config sets them in an environment's `harness_options`, on the command line like any key
  (`environments.0.harness_options.num_retries=5`), and for evaluation rollouts only in `eval_overrides`. They are
  checked before anything runs, an unknown or badly typed one is a configuration error (exit 78) that names its
  key, and `lock.json` records the options each harness ran with, defaults included. `openhands` takes
  `reasoning_replay`, `request_timeout_s`, `num_retries` and `condenser`: its defaults train, and Harbor's
  `openhands-sdk` adapter's settings are an evaluation's options, from the same toolchain
  ([RFC 0016](docs/rfcs/0016-component-configuration.md)).
- **Harnesses inside task sandboxes.** A command-line harness runs in a task's own sandbox, also one without a
  network: the gateway serves each rollout on a Unix socket of its own, which a route relays into the sandbox, and
  the harness runs from a toolchain built once per site (`platoon toolchain build`), under `toolchains` in the site
  profile's cache, else `~/.cache/platoon/toolchains`, and mounted read-only; a run never builds one, and refuses to
  start, naming the `platoon prepare` command, while it is missing. A spec's `[budget] enforce = "gateway"` caps its
  policy calls at the gateway, a harness in a task sandbox may make only policy calls, `on_harness_exit` says whether a
  harness that ran out of time or crashed is graded or left out, and `rollout_config.sampling_precedence: config` has
  policy calls sample as the config says, never as a harness asks.
- **A recording model gateway.** Every model call, from Python (`ctx.chat`) or from any client over
  OpenAI-compatible HTTP, is rendered, sampled, parsed and recorded with its exact tokens. The OpenAI Responses and
  Anthropic Messages dialects are experimental. An MCP tool plane serves each agent the environment's tools. A
  sampling call fails after the sampling timeout, which a config sets beside its model
  (`train.sample_timeout_seconds`, `inference.sample_timeout_seconds`); without the key, `TINKER_SAMPLE_TIMEOUT_SECONDS`
  sets it, else 600 s. The key wins over the variable, which a gateway job may export for every client it runs, a
  training run's rollout workers sample under the trainer's bound, and `platoon train` and `platoon eval` log the
  bound with where it came from ([The sampling timeout](docs/how-to/train-and-eval.md#the-sampling-timeout)).
- **Evaluation with hosted models.** `platoon eval` and `platoon rollout` can evaluate a model behind any server
  that speaks the harness's own dialect: `inference.hosted` names the base URL and the variable that holds its key,
  and `inference.model_name` the model. The gateway forwards each call unchanged and treats tools as training does,
  so trajectories, step budgets and usage are still recorded, and the key never reaches the agent's sandbox.
  `inference.judges: false` turns off the model calls that judge agents rather than solve the task
  ([Evaluate a hosted model](docs/how-to/evaluate-hosted-models.md)).
- **Tool calls for any chat template.** The `chat_template` renderer renders with the model's own template and
  reads tool calls with a parser chosen by name, by a round trip through the template (`tool_call_parser: auto`),
  or written by you (`module:attr`).
- **Environments as services:** OpenReward (ORS) environment classes, in process or on environment servers;
  Harbor task directories, graded by their own tests in `process`, Docker, Singularity/Apptainer or enroot
  sandboxes (enroot takes the Singularity provider's option names); step environments through `pt.gym`; and task
  tables. An environment entry's `exclude_tasks` leaves out tasks that cannot be solved or graded as written, and
  `lock.json` records which.
- **Every training task trains.** A single environment's tasks are drawn in seeded shuffled passes that go on from one
  epoch to the next: an epoch holds the whole batches its tasks fill, and the tasks left over start the next epoch
  instead of being left out of every one. `environments` may also list several entries, each with its own service,
  harness, program and options. Training interleaves their tasks in a weighted fair order: `weight` is `tasks`, the
  default, each entry's number of training tasks, which draws every task of the mixture equally often, or a number,
  and the entries share the slots in proportion. The epochs hold as many tasks as the entries do, filled to whole
  batches, each going on with the order where the one before it stopped, so an entry that gets fewer slots than it has
  tasks draws the rest in the next epochs, not the same ones every epoch. A resumed run draws the batches the run it
  resumes would have drawn. `platoon train` logs each entry's share and how often each of its tasks is drawn over the
  planned steps, and warns when the weights leave an entry's tasks out of the run or draw one entry's tasks more than
  4 times as often as another's; `platoon config validate --tasks` lists the tasks and prints the same before a run
  ([Train on several environments](docs/how-to/train-and-eval.md#train-on-several-environments)).
- **Harbor datasets as published.** A Harbor environment names a dataset (`hf://org/name@commit`, or a local
  directory) that `platoon harbor fetch` downloads once into a content-addressed cache and pins with a lock, which
  rollout workers read offline; `platoon harbor images` builds the tasks' images once per site from their
  Dockerfiles, sharing one image among tasks whose build steps match, and an entry that names no image map uses
  `images/<provider>.json` in its cache, unless it serves a directory. Tasks may sit at any depth under a registry,
  a separate verifier grades a task's artifacts in a sandbox of its own, and a dataset's reward contract says which
  of its rewards trains. A dataset whose tasks report that reward under different names is one environment: the
  contract lists the names (`reward: [reward, correctness]`), and each reading must report exactly one of them, from
  one reward file; a group whose rollouts trained different names does not train (`reward_key_mismatch`). Each
  rollout's record names the reward key it trained (`misc.outcome.reward_key`), each step counts its rollouts by it
  (`train/batch/reward_key/<name>`), and `platoon eval`'s report by `reward_key_counts`. A fetch records index
  fields from the dataset's files and, with `index: [{metadata: [...]}]`, from each task's `task.toml` `[metadata]`
  table, records what each task's sessions upload and the digest of each of its input files, and refuses tasks that
  upload more than `refuse.uploads_over_mb`. `holdout` holds out whole groups of a dataset's tasks for evaluation, a
  share of each source, and lists them as the `eval` split and the rest
  as `train`, with a report of each split's makeup; tasks that share an input file (a text file compared as text,
  whatever its line endings) are always in one group, so no held-out task holds data that training has seen, and
  `holdout.ignore_inputs` names files that every task holds by design
  ([One dataset, one environment](docs/how-to/sandboxes-and-harbor.md#one-dataset-one-environment)). A run's
  dataset has each task's index fields as columns, as text, which `train.label_by` can name; a field named like
  one of platoon's own columns is left out, and a configuration error when `label_by` names it.
  `platoon harbor fetch`, `verify` and `images plan|build|verify|alias` take a training or evaluation config, with
  overrides, and act on each Harbor entry's dataset as the run would, `images` with the entry's image map and its
  sandbox provider's builder; `fetch` leaves a dataset pinned to a commit alone when its lock is current (`--repair`
  fetches anyway), and fetches a lock written before input files were recorded once more. Fetches of one lock, and
  builds of one image, take turns on file locks, also on several nodes, so each runs once. Fetch config files are
  deprecated: an entry of the run config says the same. A build step that fetches remote content, an `ADD` of a URL
  or of a git repository, also one whose URL an `ARG` or `ENV` value holds, is built as Docker builds it: the `oci`
  builder runs it with BuildKit, which verifies its `--checksum` and does not extract a remote archive, the build key
  covers it as written, and the `def` builder, which cannot fetch, refuses it and names `oci`. The `oci` builder also
  builds a Dockerfile whose `# syntax=` directive names a 1.x release of Docker's Dockerfile frontend, with the one
  built into SingularityCE's BuildKit, which cannot load a frontend image; it refuses any other frontend.
- **One command prepares a config.** `platoon prepare CONFIG` makes, once per site, what the config's runs need and
  never make themselves: each Harbor dataset fetched into its lock, the images its tasks build, with the builder of
  its sandbox provider, into its image map, its harnesses' toolchains, and the policy's tokenizer, into default
  locations under the site profile's cache, else `~/.cache/platoon`. Each step does nothing once current, two prepares
  at once take turns, also on two nodes, and `--dry-run` says what is missing
  ([Prepare a config](docs/how-to/sandboxes-and-harbor.md#prepare-a-config-once-per-site)). It never replaces a
  toolchain built from another recipe, which a run of another config may be running from: the step names the forced
  build that does. Before any rollout, `platoon train`, `eval` and `rollout` refuse (exit 78) a config whose toolchains,
  or whose fetched tasks' images, are not built, naming the `platoon prepare` command for their config, overrides
  included. On a laptop or a single node, `--prepare` on any of them runs that command first, with its lines on stderr,
  and starts the run only once the config is ready, else exits with prepare's status; on a cluster, prepare once on a
  CPU node instead, since in a GPU job the flag spends GPU time building.
- **One run per run directory.** A run's first start writes `identity.json`: its model, renderer and held-out
  tasks. A resume that would change them stops with exit 78 before it opens the training service, and records why in
  `lock.json`; training that evaluates warns when its evaluation lists training tasks
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). `platoon eval` writes each start to
  a new directory, `runs/eval/<policy>/<UTC time>` (the checkpoint, else the model), unless `inference.output_dir` names
  one, and never resumes another policy's rollouts: a directory whose rollouts another model, checkpoint, renderer or
  hosted base URL made, or with judges switched, stops it with exit 78 before anything in the directory changes
  ([Evaluate](docs/how-to/train-and-eval.md#3-evaluate)).
- **Cadences in training steps, concurrency in rollouts.** `eval.every` evaluates the eval split every that many
  training steps, and `checkpoint.every` saves a permanent checkpoint as often; 0, the default of both, does neither,
  while the launch, recovery and final checkpoints are saved either way
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)). Each evaluation logs the step it
  started at (`eval/step`), and the first is at step `every`, as in the original platoon: `eval.at_start: true`, false
  by default, also evaluates the starting policy, at step 0, and holds the first update until that evaluation ends, so
  that the held-out curve starts at the base model
  ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)). A start that resumes, by hand or
  after a failure, evaluates at the multiples of `every` from the step it resumes at, and that step only if the run
  directory's `evaluations.jsonl`, which records each evaluation that completed, does not list it: that step's
  evaluation runs again if a crash cut it short, and does not run twice
  ([Evaluation in a resumed run](docs/how-to/train-and-eval.md#evaluation-in-a-resumed-run)).
  `train.max_concurrent_rollouts` and `eval.max_concurrent_rollouts` count the rollouts that run at once, a whole group
  at a time: by default every rollout of a batch in training, and 256 in an evaluation. **Breaking:** an `eval` or
  `checkpoint` section that sets neither `every` nor the deprecated `strategy` no longer evaluates, or saves a permanent
  checkpoint, each epoch, and `every` alone no longer counts epochs: set it in steps (an epoch is the training tasks
  over `train.batch_size`). An evaluation's default concurrency counts 256 rollouts, not 256 groups, which differs once
  its `group_size` is above 1.
- **Evaluation that rolls out as training does.** A training config's `eval.workflow_config` is laid over
  `train.workflow_config`: an evaluation during training keeps training's rollout settings (`rollout_config`: step
  and time limits, sampling, `extra`) and rollout-worker settings (`rollout_isolation`, `worker_start_method` and the
  other `worker_*` keys) unless it sets them, while the keys that shape groups, advantages and filters keep
  evaluation's defaults (one rollout per task, no filters, no straggler cutoff). A key it sets replaces training's;
  `rollout_config` and its `inference_params` merge key by key, `null` unsets a value training sets, and `extra`
  replaces training's whole ([Evaluate during training](docs/how-to/train-and-eval.md#evaluate-during-training)).
  **Breaking:** an evaluation that relied on evaluation's defaults for these settings (the harness's own step limit,
  no time limit, the gateway's sampling and 4,096-token completions, in-process rollouts) now takes training's; set a
  key under `eval.workflow_config` to keep the old value.
- **Harbor grading that agents cannot forge.** Grading first kills every process the agents left running
  (`Sandbox.kill_processes`, part of the sandbox provider contract). Where the tests run in the sandbox the agents
  worked in, it makes the tests' and the reward's directories afresh; a separate verifier gets the task's artifacts
  as directories and regular files only. In either, grading kills what the tests left running and reads only
  regular reward files. A reward file of any other kind scores 0, and so does an attempt whose agents removed,
  moved or locked their workdir (`workdir_missing`), without running the tests; a grade whose processes cannot all
  be killed is an untrainable error
  ([How grading keeps the agent out](docs/how-to/harbor-grading.md#how-grading-keeps-the-agent-out)).
- **Nothing a stopped rollout started keeps running.** Rollout workers label the sandboxes they start, and the
  sandboxes a killed or hung worker left are reaped by label. Every sandbox of a run also carries the run's
  directory as its `platoon.run` label, so a job can reap what its own run left and no other run's. A rollout's
  deadline counts its sandbox starts and its grading, so a slow verifier does not kill the worker that grades it.
  Without a `rollout_config.timeout`, the deadline counts the harness's own time limit where the environment says
  what a rollout needs beyond it, as Harbor's does: the earliest of its spec's `timeout_s` and a Harbor task's
  `[agent] timeout_sec` (times `agent_timeout_multiplier`). So a Harbor config needs no `timeout` to bound its
  rollouts, and an evaluation's worker processes (`use_subprocesses`) let a harness run as long as its task allows.
- **Rollout workers that start in seconds.** With `rollout_isolation: process`, a rollout's worker is forked from a
  fork server that imported the worker's stack and built the run's tokenizer and renderer once
  (`worker_start_method: forkserver`, the default), instead of a new interpreter that imports it (`exec`: about 20 s,
  a minute when 64 start at once on one node). A forked worker is the process an exec'd one would be, and one the
  fork server cannot fork starts by exec. On GPU nodes too: neither the fork server nor a worker sees a GPU or starts
  the CUDA driver, whose thread would keep the fork server from forking. The singularity provider's
  `rootfs_layout: fuse-overlay` (hpc-sandbox 0.3) starts a sandbox in about a second over one extraction of its image
  per node, and a run that sets a layout its provider cannot use is refused when its tasks are listed
  ([The singularity provider's root filesystem](docs/how-to/sandboxes-and-harbor.md#the-singularity-providers-root-filesystem)).
- **Runs that start again by themselves.** With `watchdog.max_restarts: N`, `platoon train` trains in a child process
  and starts it again from the newest checkpoint after any failure but a configuration error (exit 78) and an
  interrupt: the watchdog's exit, a crash, a training service that stalls or dies. It stops after `N` restarts, or after
  three starts in a row that saved no new checkpoint. A start asks the training service for its training client until
  the service accepts it, for up to 20 minutes, logging each refusal, so no delay is needed while a service still holds
  a dead start's session. `platoon train` reaps the sandboxes labelled with its run when it starts, before each restart
  and when it exits, `platoon eval` those labelled with its output directory when it exits, and `platoon sandbox reap
  RUN_DIR` those of a run that was killed; each logs how many. A run directory records the process that runs it
  (`process.json`): `platoon sandbox reap` refuses a run that still runs, and `platoon train` and `platoon eval` a
  directory that another of their processes runs. `python -m platoon.train.tinker.restart_wrapper`, which restarted on the watchdog's exit
  only, is deprecated, and 0.3.0 removes it ([Long runs](docs/how-to/train-and-eval.md#long-runs)).
- **Subagents.** An agent starts subagents with `ctx.agents.spawn`, under step budgets and depth limits, with
  forked environment state and optional verifiers. One rollout trains every agent in the tree, and episode
  programs (`@pt.program`) compose episodes and shape rewards. The default program, `single`, gives subagents the
  root's success when `propagate_root_success` is set (after scoring them 0 when `skip_subagent_reward_computation`
  is), and `platoon-delegation` is a reward processor for recursive agents.
- **Outcomes that say nothing about the policy** (a crashed grader, a dead sandbox, a timeout) never train as a
  reward of 0, and do not count toward a group's `min_successful_group_size`.
- **A Tinker key that is missing or refused is one line.** `platoon train`, `eval` and `rollout` check
  `TINKER_API_KEY` before they prepare, write or reap anything, and without a key that starts with `tml-` stop with
  one line and exit 78; a hosted model needs the variable its `api_key_env` names instead, checked as early, and a
  report, or an evaluation or rollout that serves no policy, needs neither. A key that the Tinker service refuses
  (HTTP 401, or 403 for what the key may not do) ends each of them with one line that names the key and the service,
  instead of the client's traceback, and exit 78: `platoon eval` starts no rollout after it, and a supervised
  `platoon train` does not restart.
- **Recovery checkpoints that do not fill the disk.** With `checkpoint.keep_recovery: N`, training deletes its
  recovery checkpoints beyond the newest N through the Tinker API, beside training and on a Tinker session of its
  own: never a permanent checkpoint, the one a resume starts from, one that rollouts may still sample from (within
  `train.max_staleness` policy versions) or another run's. A config whose `keep_recovery × recovery_every` does not
  exceed `max_staleness` does not load (exit 78). A resumed run goes on where it left off
  (`checkpoints_deleted.jsonl`); failed deletes are counted (`checkpoint/delete_failures`), and three in a row, or a
  service that cannot delete checkpoints, stop the deletes with a warning while training goes on
  ([Checkpoints and resume](docs/how-to/train-and-eval.md#checkpoints-and-resume)).
- **The same training however an environment runs.** A tool's failure is graded alike in process and on an
  environment server, a served class named by `service_kwargs.env_class` shows agents the same tools as in process,
  and a test plays both modes and compares every token, outcome and datum.
- **Extensions.** Installed packages provide harnesses, episode programs, named environments and reward processors
  by name through entry points, which platoon imports only when a config names them; `platoon extensions list`,
  `show` and `check` list them, describe one and check them ([Extending platoon](docs/extending/index.md)). Each
  ships packages of its own: `platoon` is a regular package, which no extension installs into. The repository's
  first-party extensions (`extensions/`) are built and tested with core, in one uv workspace.
- **`platoon.testing`** runs environments, harnesses and programs on CPU with a scripted policy and a toy
  tokenizer, with no model download.
- **One command line:** `platoon train`, `eval`, `rollout`, `config`, `harness`, `program`, `env`, `extensions`,
  `harbor`, `toolchain` and `viz`, with strict configs that build on each other through `extends:` and load as the class
  their environments' `trainer_config` names, so that a plugin's config sections validate, and a `lock.json` in every
  run directory with the versions of platoon and its main dependencies and the commit of each one installed from git,
  and the images the job pulled for its sandboxes when it names their record
  (`PLATOON_IMAGE_PULLS`). Before anything trains, a run refuses (exit 78) a config whose rollouts could never
  train, such as a TOML harness spec whose settings never apply or whose tools never reach the task, or environment
  grading that cannot grade a subagent's own goal, and one whose `trainer_config` does not resolve or one of whose
  `package` modules does not import. `platoon config validate` resolves every component a config names, as a run
  does when it starts (`--no-resolve` checks the files and named environments alone), takes the run's `key=value`
  overrides, and checks a file of environments alone against the entry schema. An entry's `service_kwargs` are bound
  to what its service takes when the config loads, so a misspelled option fails with a did-you-mean, and a value the
  service refuses when the run builds it is a configuration error that names the entry.
- **A site profile.** Where a site keeps platoon's files and which runtime starts its sandboxes go in one file per
  site, never in a project: `~/.config/platoon/site.yaml`, or the file `PLATOON_SITE` names, with four keys
  ([Write the site's profile](docs/how-to/clusters.md#4-write-the-sites-profile-once)). `cache` is the root of what
  `platoon prepare` makes (`<cache>/harbor`, `<cache>/toolchains`); `runs` holds the run directory of a training
  config without `log_path` (`<runs>/<experiment>/<trial>`, the experiment named after the config file unless it
  names one, a directory that belongs to that file, so that another project's file of the same name is refused rather
  than take over its run) and the directory of each start of an evaluation without `inference.output_dir`
  (`<runs>/eval/<policy>/<UTC time>`), while a relative `log_path` a config names stays relative to the working
  directory; `provider` and `provider_options` start the sandboxes of every Harbor entry that names no provider, under
  the entry's own `provider_options`, and of `platoon sandbox reap` for a run whose record names none. The profile
  cannot weaken isolation: it names only a provider whose class says it isolates its sandboxes (`docker`, `singularity`,
  `enroot`; never `process`) and only the options that keep that isolation (`allow_network_none`, `rootfs_layout`,
  `scratch_root`, `image_cache`, `stage_images`, `gpu`), and `platoon train`, `eval` and `rollout` refuse a Harbor entry
  on an unisolated provider that the run config does not name itself. `~` and `$VAR` expand in it, and a key it does not
  take, or a `PLATOON_SITE` that names nothing, is a configuration error. `platoon train`, `eval`, `rollout` and
  `prepare` print the profile they used and what it gave the config, and refuse a provider from it that the node cannot
  run, quoting it. `config.resolved.yaml` holds what it gave and names each Harbor entry's provider, so a run directory
  loads the same without it, and `lock.json` records the profile's path, SHA-256 and values (`site`) and each Harbor
  entry's provider and options with where each came from (`sandbox_providers`). A training config without an `eval`
  section does not evaluate. platoon's temporary files (rollout workers' files, gateway sockets, sandbox and harness
  state, artifacts and staged toolchains) go to node-local scratch, the first of `$PLATOON_NODE_SCRATCH`,
  `$PBS_LOCALDIR`, `$SLURM_TMPDIR` and `$TMPDIR` that is set, else `/tmp`, rather than `$TMPDIR` alone.
- **Paths relative to the file that names them.** A config file's input paths are relative to that file, as
  `extends:` is: an environment's harness `.toml` spec and the options its service reads files from (a Harbor entry's
  `dataset`, `dataset_dir`, `task_dirs`, `splits`, `lock`, `tasks`, `overlays` and `mounts[].source`, a task table's
  `path`), so a config runs the same from any directory. A split manifest named relative to a `dataset_dir` that the
  same file names stays relative to it. An override's paths and the outputs (`log_path`, `inference.output_dir`, where a
  leading `~` names the home directory) stay relative to the working directory, and `config.resolved.yaml` holds them
  absolute. **Breaking:** these input paths resolved against the working directory before, and nothing falls back to it:
  a relative input that names nothing beside its file stops the command with exit 78, and names the path relative to the
  working directory too when that one exists ([Write and override
  configs](docs/how-to/configs.md#2-build-on-it-with-extends)).
- **Deprecations that say what replaces them.** A config key that is going away keeps working for one minor release,
  with a `pt.PlatoonDeprecationWarning` (a `FutureWarning`, which Python shows) that names the key's path, its
  replacement and the release that removes it. The commands name each one once on stderr, `lock.json` lists them
  (`deprecations`), and `platoon config validate` lists them under each file and, with `--strict`, fails on them. The
  first is `rollout_config.propogate_root_success`, the misspelling of `propagate_root_success`, which 0.3.0 removes,
  and `platoon harbor` warns the same way of a fetch config file, which 0.3.0 removes too.
  0.3.0 also removes `train.rollout_group_key`, now `train.label_by`, and `workflow_config.straggler_policy_key`,
  whose straggler policies key on `label_by`'s labels now; a workflow config that still sets it keys its own on that
  column until then. It removes the `strategy` of `eval` and `checkpoint` as well: `every` alone counts training
  steps, while a config that still sets `strategy` keeps its meaning until then (`epoch` counts `every` in passes over
  the training tasks, `none` turns the event off, and an `every` left out is 1, as it was). And it removes
  `num_concurrent_rollout_workflow_workers` of `train` and `eval`, which counted rollout groups: until then it reads
  as that many groups' rollouts, so `max_concurrent_rollouts` is the old value times the section's
  `workflow_config.group_size`. It removes a Harbor entry's `cache`, `image_map` and `toolchain_root`, and the
  variables `PLATOON_HARBOR_CACHE` and `PLATOON_TOOLCHAINS`, which the site profile's `cache` replaces: until then they
  work as before, and a variable set beside a profile's `cache` must name the same directory.

### Install

```bash
pip install 'platoon[tinker,ors]'
```

- Python 3.12 or later, on Linux; CI tests 3.12, 3.13 and 3.14. On 3.14, platoon needs datasets 4.4 and dill 0.4
  or later, whose picklers can hash a dataset there (pip upgrades older ones), and importing the Tinker SDK warns that
  pydantic's V1 core is not compatible with 3.14: the SDK takes only V1's date parsers and typing helpers, with which
  it builds its responses as on 3.12.
- The `tinker` extra pins the Tinker SDK to 0.16.1 and takes tinker-cookbook 0.3: later tinker-cookbook releases
  change what a renderer's parse returns or need a later Tinker SDK, which a future release adopts.
- The `hpc` extra installs hpc-sandbox 0.3 from PyPI for the `singularity` and `enroot` sandboxes. It deletes a
  stopped sandbox's files in the background, and the next sweep on the node deletes what the end of a job cut short;
  where the sandboxes' scratch outlives jobs, start each job with hpc-sandbox's janitor
  ([Sandboxes left by stopped rollouts](docs/how-to/sandboxes-and-harbor.md#sandboxes-left-by-stopped-rollouts)).
- Number search, the first run's environment, ships in the package as the built-in named environment
  `platoon-number-search@1` (`platoon.envs.number_search`), which generates its tasks from a seed. The recipes and
  the runnable examples are in the repository, not in the package: clone it to follow the tutorials. TextCraft is
  the first-party extension `platoon-textcraft` (`extensions/textcraft`), a distribution with a version of its own
  (`pip install 'platoon-textcraft[platoon]'`), which also exports TextCraft-Synth as a Harbor dataset
  (`python -m platoon_textcraft.harbor export`), whose verifier replays the game's action log. `recipes/rao` carries
  the RAO paper's TextCraft-Synth arms over to Tinker: its single agent (`linear.yaml`), its recursive agent
  (`recursive.yaml`: 25 steps for every agent, at most 6 levels of delegation, and each subagent rewarded for
  crafting its own targets, the paper's Dense rewards) and its Sparse ablation (`ablation_sparse.yaml`: every
  subagent takes the root's success). A recursive agent that sizes its children's budgets from its own 200 steps is
  not one of the paper's arms, so the recipe has none.
- `platoon.__version__` and `platoon --version` give the installed version, and the package ships type information
  (`py.typed`).

### Documentation

[The documentation](docs/README.md) has a getting-started guide, tutorials whose examples a CPU test runs, how-to
guides, guides to extending platoon, a command-line, config and Python API reference generated from the code, an
architecture overview and the RFCs, among them the drafts on Harbor datasets, images, grading and multi-container
tasks (0010) and on the public surface and conformance (0017), and the accepted design for component configuration
(0016).
[CONTRIBUTING.md](CONTRIBUTING.md) covers development, Conventional Commits, the checks CI runs, among them the
snapshot of the public API and the import boundaries of core, the first-party extensions and the examples, and how
to cut a release.
