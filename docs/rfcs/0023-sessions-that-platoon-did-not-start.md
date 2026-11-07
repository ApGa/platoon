# RFC 0023: Sessions that platoon did not start

| | |
|---|---|
| Status | Draft |
| Created | 2026-11-07 |
| Authors | Apurva Gandhi (@ApGa) |
| Builds on | [RFC 0001](0001-native-multi-agent.md) (`RewardEvent`, built here), [RFC 0005](0005-gateway-dialects-and-tito.md) (dialects, call kinds, `retry_key`), [RFC 0007](0007-records-and-atif.md) (its record layout, applied to sessions), [RFC 0015](0015-hosted-models-and-judges-in-eval.md) (hosted models, which never train, and which this RFC extends to a shadow and to judges of sessions), [RFC 0021](0021-trainers-credit-and-policies.md) (the trainer lifecycle and its SFT trainer's shape, credit, versions, the service needs U1-U4 and U10), [RFC 0022](0022-programs-that-run-rollouts.md) (`running` and `roles`, its rule for root roles, its judge-mint refusal, frozen hosted judges, pure rewards over evidence, the alarms of its archive hook, its toolkit's seeds); the gateway (`platoon/gateway/core.py`, `platoon/gateway/server.py`, `platoon/gateway/hosted.py`), the `sft` trainer (`platoon/train/trainer/sft.py`), the converter (`platoon/utils/tinker_data_processing.py`) and the credit API (`platoon/train/credit/api.py`) |
| Discussion | The pull request that adds this RFC |
| Implementation | None yet. Nothing here is built before a product or harness with traffic exists (PO-0); the cheapest first step, logged sessions as environment-generator seeds, needs RFC 0022's pieces and about 100-150 lines more (section 10) |

## Summary

Platoon trains on rollouts it starts. Every training group is a draw of the dispatcher, a rollout's gateway lives as
long as its rollout, and a resume draws again whatever was in flight. A product's sessions are different: its users
start them, nobody can run them again, and their rewards arrive late, in part, or never. This RFC designs online RL
from such sessions, whether a product's users or a developer's own harness start them, around one principle:
**serve, journal, resolve, then train.** A *session* here is one conversation or working session that a product or a
harness opens against platoon; a *Tinker session* is a client's connection to the training service, and is always
called so.

- **`platoon serve`** is one standing gateway process with sessions, keys minted per session, receipts (the call ids
  the gateway already returns) and a feedback route. It journals every call before it answers, admits sessions by
  serving capacity alone, and samples through a Tinker session of its own that loads the active checkpoint.
- **The session journal** holds RFC 0007's per-rollout events and token sidecar, plus a request record per call, one
  directory per session, so that deleting a session removes a directory.
- **The resolver** completes each session's evidence outside the trainer. It waits for the feedback and judge verdicts
  the session's source declares, up to a deadline, and a pure reward scores the evidence. Missing evidence is
  censoring, never an implicit 0.
- **The `sessions` trainer** (`trainer: sessions`) reads resolved sessions in journal order, converts them with
  today's converter, credits them with RFC 0022's `running`, drops stale rows and steps with `cispo`, as `sft` steps
  through a dataset. It holds no pending state, rolls nothing out, and resumes from a cursor: nothing runs again and
  nothing is lost.
- **Governance.** An activation record, not the training head, chooses what serves, behind a `platoon eval` gate, with
  rollback, per-tenant adapters, capture filters and behaviour monitors that stop training.
- **Compatibility.** `platoon serve` and `trainer: sessions` are new and chosen by name: a config that uses neither
  writes today's bytes, and the golden trace gates every pull request. The plan starts with seeds, logged sessions
  rebuilt as tasks by an environment generator, and builds PO-0 to PO-4 only once a product with traffic exists.

*The trainer never waits for a user:* evidence resolves in the journal, and the trainer reads only what has resolved.

## Motivation

"Online RL" names three things, and they need different machinery.

| | Harnessed RL | Product RL | Deployment-loop retraining |
|---|---|---|---|
| Who starts an episode | the trainer, from a dataset row | a user | a user; the logs are used later |
| Can it run again? | yes, in a fresh sandbox | no; at most its prompt can | its prompt, offline |
| Reward | a verifier, tests, a judge | a user's signal or the next state: late, partial or never | a reward model, a judge or logged preferences |
| Samples per prompt | a group of G | one | G by sampling the prompt again, or logged accepted and rejected pairs |
| Cadence | minutes per step | hours per deployment: 1.5-2 h for Cursor Tab, about 5 h for Composer's real-time RL | a day or more |
| Examples | platoon, [Polar](https://arxiv.org/abs/2605.24220) | Cursor Tab and Composer's real-time RL, Trinity-RFT's serve mode, OpenClaw-RL's personal agents, Reef, AReaL's online mode (all under Prior work) | retraining on API or editor logs |

Platoon does the first. Product RL adds four needs: episodes that start outside the trainer, rewards that arrive
late, credit for single samples, and the durability and governance a product requires.

### Cases

| | Case | What stops it today |
|---|---|---|
| C1 | **A product's users.** An assistant whose users accept, reject, report or follow up, minutes to days later or never | no ingress, no feedback route, nothing durable; a group of one trains nothing |
| C2 | **A developer's own harness sessions.** Claude Code or Codex pointed at platoon with an open-weight policy, every working session a sample | the gateway lives only inside a rollout; sessions a hosted model answered hold no tokens and never train; training through Claude Code waits on a terms-of-use question (RFC 0005's open question 1) |
| C3 | **Next-state judging**, as OpenClaw-RL does it: a frozen judge scores each turn by the reply or tool result that followed | no judge outside an episode, no credit per step |
| C4 | **A shadow:** serve the product's current model and record, to measure feedback and judges before anything trains | the hosted pass-through records replies, tool calls and usage, not requests (`platoon/gateway/hosted.py`) |
| C5 | **Logged sessions as training tasks:** an environment generator rebuilds a recorded session as a Harbor task | RFC 0022's seeds come from papers, repositories and documents, none from a session, and no audit checks a task's faithfulness to one |
| C6 | today's runs | they must not change at all |

### What platoon has, and what it lacks

**The capture half is built, for rollouts platoon launches.** One gateway speaks OpenAI Chat, OpenAI Responses and
Anthropic Messages to unmodified harnesses (`platoon/gateway/server.py`). Each call records its exact prompt ids,
sampled ids and logprobs, and the policy and version it sampled (`TinkerLLMInteraction`, `platoon/gateway/core.py`).
Keys are minted per agent, and per subagent for clients that mint them; each call's kind is recorded as the client
labels it; keepalives hold clients through weight pushes; trajectories merge only on exact token prefixes; no
re-rendered token trains. This capture is comparable to AReaL 2.0's and stricter than OpenClaw-RL's or Reef's, which
train re-rendered tokens.

**The control half is missing:** everything a product needs once it starts the episode.

| A session platoon did not start needs | Platoon today |
|---|---|
| a standing ingress: sessions, keys per session, admission apart from training | a gateway per rollout, on loopback, that takes only its rollout's keys (`EmbeddedGateway`, `platoon/harness/rollout.py`) |
| receipts and late feedback | each reply already carries its call id, as the response's `id` in every dialect and in `x-platoon-call-id` when it is not streamed; no feedback route exists, and RFC 0001's `RewardEvent` is designed, not built |
| durability at capture | "Tokens are never persisted" (RFC 0007); a resume draws outstanding work again (`platoon/train/checkpoints/dispatch_state.py`), and a session cannot be drawn again |
| an input the trainer does not schedule | every group is a draw: a start opens one `RolloutSource`, once (`platoon/train/trainer/run.py`) |
| credit for single samples | a group of one "trains nothing" under group baselines (`platoon/train/entry_settings.py`); RFC 0022's `running` is built but not merged |
| credit per step | advantages per token (`TrainingSample.set_advantages`), but one reward per agent and no call spans in merged datums (`platoon/train/credit/api.py`) |
| versions that name one sampler | after a crash, version numbers are reused (RFC 0021, rule 6), so a row from a lost sampler misreads its age |
| serving that outlives the trainer | rollout workers follow the trainer's sampler through a `FollowingSamplingClient`, which borrows the trainer's own Tinker session (`platoon/train/workers/binding.py`); a step that saves no full checkpoint publishes an ephemeral sampler, which has no path for another Tinker session to load (`platoon/train/policy/checkpoints.py`); on miles-tinker a Tinker session's release unloads the samplers it saved (RFC 0021's U10) |
| serving governance | evaluation during `rl` training only; no activation, gate or rollback |
| data governance | the hosted key's hygiene only (`platoon/gateway/hosted.py`); no capture filter, no deletion |
| clients platoon does not configure | an untagged call trains as a `policy` call; `response_format`, penalties, `logit_bias` and stateful Responses fields are ignored silently (`platoon/gateway/dialects/responses.py`); sampling parameters are not recorded; a client that disconnects cancels its call, which leaves no record (`platoon/gateway/server.py`); RFC 0005's `retry_key` is not built; the Responses and Messages dialects are experimental |

### Prior work

| System | What it is | Strongest part | Weakest part |
|---|---|---|---|
| [AReaL 2.0](https://arxiv.org/abs/2607.01120) | a position paper without experiments, and a refactor into services; its online proxy (`/rl/start_session`, `/rl/set_reward`, weights reloaded after every step) predates 2.0 | capture on the server with a version per token; persistent sessions | it refuses groups online, and its v2.0.0 example zeroed every reward; sessions live in memory (`deque(maxlen=1024)`); nothing is evaluated online |
| [OpenClaw-RL](https://arxiv.org/abs/2603.10165), personal track | a URL swap and two headers; a frozen judge votes on each action and the state that followed; hindsight hints drive on-policy distillation | no dataset or verifier | it trains a re-rendered, re-tokenized turn, cutting or zero-padding the sampler's logprobs to fit (OpenClaw-RL: `openclaw-rl/openclaw_api_server.py`); no versions; one key for every user |
| [Reef](https://github.com/Human-Agent-Society/reef) | a serving gateway with receipts, a report API, durable records, recipes that turn records into samples, bounded training jobs and activation rules | the only open system that pairs durable, at-most-once feedback with serving chosen by activation rules | served calls record no tokens since [its RFC 1054](https://github.com/Human-Agent-Society/reef/issues/1054), so training renders each record again with the chat template; a trajectory takes its last call's version (Reef: `reef/core/job_data.py`) |
| [Trinity-RFT](https://github.com/agentscope-ai/Trinity-RFT)'s serve mode | the smallest open product loop: a version per response, `POST /feedback {msg_ids, reward}`, SQL rows with a `consumed` flag, replicas updated one at a time | a durable join of late rewards (Trinity-RFT: `trinity/explorer/proxy/recorder.py`) | OpenAI Chat only; each call a sample, with no session or multi-turn merge; no authentication |
| Cursor: [Tab](https://cursor.com/blog/tab-rl), [Composer's real-time RL](https://cursor.com/blog/real-time-rl-for-composer) | policy gradient on live traffic, kept on-policy by deploying a checkpoint every 1.5-2 h (Tab) or about 5 h (Composer, after evaluation suites) | rewards with offsets that give a lone sample a sign (Tab: 0.75 accepted, −0.25 rejected, 0 not shown) | a product's own stack; the reward hacks it reports (below) |

Capture behind a provider URL, prefix merging, truncated importance sampling, distillation from hints and judges of
single steps are not new, and platoon has or plans most of them. The lessons others paid for, and the rule each
becomes here:

| Failure | Where | Rule here |
|---|---|---|
| centering a group of one zeroed every reward; the fix was a warning | AReaL's v2.0.0 example, [issue 1473](https://github.com/areal-project/AReaL/issues/1473), [fix 1474](https://github.com/areal-project/AReaL/pull/1474) | the `sessions` trainer refuses a group-centering estimator on sessions that resolve alone (section 7) |
| unrelated sessions grouped for GRPO | AReaL, [issue 1304](https://github.com/areal-project/AReaL/issues/1304) | sessions group only by a prompt key their source declares, never by arrival (section 5) |
| a missing reward scored as 0 | AReaL's first online mode | S4: missing evidence is censoring |
| re-rendered tokens trained; a trajectory versioned by its last call | OpenClaw-RL; Reef | S5: sampled ids only, stamped with the oldest call's version, as platoon does today |
| refusals (429, 503) driven by training; queues in memory; records purged at each weight push | AReaL; OpenClaw-RL | S1 and S2: serving never waits on training, and journals before anything trains |
| dropping stale or invalid samples biases training toward fast, easy ones; broken tool calls dodge failures | tinker-cookbook: `CHANGELOG.md` (its entry of 2026-10-28); Composer's real-time RL | the fresh rows of stale sessions train; drops are logged by latency; an invalid answer never scores above a failure |
| proxy optimization: sycophancy from thumbs-up rewards, "Bye! Sending Love!" closers, clarifying questions instead of edits, manipulation aimed at the few users it works on | GPT-4o; [Meta's RLUF](https://arxiv.org/abs/2505.14946); Composer; [a study of targeted manipulation](https://arxiv.org/abs/2411.02306) | rewards gated by judges with audits; monitors and stops; activation gates (section 9) |
| one call of an outside model turns a session's training data into text; prompts rewritten per harness before sampling | AReaL | S5: calls of non-trainable models are context only, and the prompt the policy samples from is never rewritten |

## Proposal

The sketches are Python 3.12 and YAML, and show shape, not final names. Every new name starts in
`platoon.experimental` or is documented as experimental (RFC 0017). EST marks an estimate and UNVERIFIED a claim
nobody has checked.

### 1. One principle, seven rules

**Serve, journal, resolve, then train.** The serving process journals each session as it captures it; a resolver
completes its evidence outside the trainer; a trainer reads only resolved sessions, in journal order. The trainer
never holds state for evidence that has not arrived, so RFC 0021's contract keeps its meaning: credit once, state
moved only in `settle`, checkpoints that resume exactly, golden traces. The update is RFC 0021's learner; what changes
is where groups come from: a journal in place of the dispatcher.

RFC 0021's rules hold, and so do RFC 0022's R2 (code the policy runs never mints a judge) and R5 (a pure function of
the evidence scores it, and one estimator credits it once). This RFC adds seven:

| Rule | Statement |
|---|---|
| S1 | **Serving never waits on training.** Admission refuses only for serving capacity, never for staleness |
| S2 | **Nothing trains that the serving process did not journal at capture** |
| S3 | **The trainer holds no pending state for external evidence.** It reads a session only once it is *resolved* (ended, past its hold, with every awaited event in or its deadline passed) and scored by a pure reward of its evidence |
| S4 | **Missing evidence is censoring**, unless the source declares otherwise (score what arrived, or give silence an explicit value); never an implicit 0 |
| S5 | **Every trained token is a sampled id with its logprob and its sampler's version,** a number never reused across starts. Nothing re-rendered or redacted trains, and calls of non-trainable models never train |
| S6 | **Product keys are untrusted.** The server mints them per session, with `policy_only`, a role allowlist and `sampling_precedence: config`; feedback takes the admin key |
| S7 | **An activation record, not the training head, chooses what serves** |

S1 and S7 hold past the end of the trainer's Tinker session (a stop, the end of its plan, a restart or a crash) only
where RFC 0021's U10 holds (section 3.4).

### 2. One session, end to end

```text
backend    POST /v1/platoon/sessions (admin key) ─▶ {session key, base URL} ─▶ the product's client
client     each call (session key) ─▶ platoon serve: render → sample (its own Tinker session) → record
             ─▶ journal: request record, L0 model_call, L3 tokens; fsync ─▶ the reply, which carries the call id
backend    POST .../feedback {event_id, name, value, call_id?, reporter} (admin key) ─▶ journal: L0 reward event
judges     a next-state judge, a hint writer ─▶ the same route ─▶ journal: L0 reward and annotation events
resolver   past its hold, every awaited event in or its deadline passed ─▶ resolved.jsonl: a reward or censored
trainer    cursor ─▶ convert ─▶ credit (running; spans) ─▶ stale check ─▶ learner (cispo) ─▶ end_step: a named
             sampler save, logged ─▶ activation: gate passed? ─▶ activation.json ─▶ platoon serve loads it
```

| # | Piece | Its job | Builds on | EST lines |
|---|---|---|---|---|
| 1 | `platoon serve` (section 3) | sessions, minted keys, receipts, feedback, admission by capacity, hygiene for clients platoon does not configure; sampling through a Tinker session of its own | `ModelGateway`, `GatewayServer`, the dialects, the trace | 1,100-1,700 |
| 2 | the session journal (section 4) | RFC 0007's events and token sidecar per session, plus a request record per call; fsynced at capture; ATIF only as an export | RFC 0007's layout and its M1-M3 | 500-800 |
| 3 | the resolver (section 5) | per source: what a session waits for, its deadline and hold, the outcome then; events applied once; judges as services | RFC 0001's `RewardEvent`; RFC 0022's pure rewards | 400-600 |
| 4 | the `sessions` trainer (section 6) | reads the journal from its cursor, converts, credits, drops stale rows, steps, publishes; waits through lulls | RFC 0021's lifecycle and learner; the shape of `sft` | 500-800 |
| 5 | credit (section 7) | `running` by default; call spans and reward events behind `needs_spans`; distillation from hints as a package estimator | the `CreditEstimator` seam | 200-300 |
| 6 | versions and staleness (section 8) | samplers saved under a name and logged; versions never reused across starts; `expired`; a datum split where its calls' versions change | RFC 0021's rule 6 and the per-datum filter | 200-350 |
| 7 | what serves (section 9) | an activation record behind a `platoon eval` gate; the pruner's pin; rollback; per-tenant adapters; capture filters; monitors that stop training | step hooks, checkpoints, the alarms of RFC 0022's archive hook | 300-500 |

About 3,200-5,000 lines of core code in all (EST), against RFC 0022's estimate of about 2,400 of core code (3,400 in
all) for its M1-M12, whose built milestones' code came to two to seven times its estimates (about four times in all).

### 3. `platoon serve`

`platoon serve CONFIG` runs one standing gateway process. It reuses the gateway as built (the dialects,
authentication, call kinds, keepalives, the budget stop, minting and the trace) and adds the session routes:

| Route | Key | What it does |
|---|---|---|
| `POST /v1/platoon/sessions` | admin | starts a session: `{source, columns, role?}` → `{session_id, api_key, base_url, expires_at}`; 429 with `Retry-After` past `serve.sessions.max_open` |
| `POST /v1/platoon/sessions/{id}/refresh`, `.../end` | admin | extends the session's idle timer; ends it |
| `POST /v1/platoon/sessions/{id}/feedback` | admin | `{event_id, name, value, call_id?, reporter}` → a reward event, `increment` at `call_id` or `terminal` for the whole session, applied once per `event_id`; `value: null` means no reward, never 0 |
| `DELETE /v1/platoon/sessions/{id}` | admin | removes the session's directory and journals a tombstone |
| the dialects; `POST /v1/platoon/agents` | the session's | today's routes; each reply carries its call id, the receipt: as the response's `id` in every dialect, and in `x-platoon-call-id` when it is not streamed |

A reward event is RFC 0001's `RewardEvent`, designed there and built here, with two changes: a new `event_id`, by
which it applies once, and its `source` field named `reporter` (`product`, `judge:<name>`), since a source in this
RFC is a kind of session (section 5). Its `agent_id` is the call's agent, or none for the whole session.

**3.1 Sessions and keys (S6).** A session is a root agent with a minted key. It ends at `end`, after
`serve.sessions.idle_timeout_seconds` without a call, or at `serve.sessions.max_seconds`.
- Its key is `policy_only`: it may make no `judge` call and, with RFC 0022's M0, mint no `judge` agent. It mints only
  the roles `serve.sessions.roles` allows, at most `serve.sessions.max_agents` agents, and its sampling follows the
  config (`sampling_precedence: config`), so the client never chooses the behaviour distribution.
- A session may name a role for its root from that allowlist, and its root samples the policy that
  `serve.agents.roles` binds its role to, by RFC 0022's rule for root roles (M3): a root of the role `policy`, the
  default, or of a role bound to none samples `serve.policy`, and no root takes `verifier` or `judge`. So a product
  can split traffic between the trained policy and a frozen Tinker one (`serve.policies`, section 12). Who gets which
  arm is the product's decision. A hosted arm beside a Tinker one waits for RFC 0022's open question 6: its M12b binds
  hosted policies to judging only.
- **Feedback takes the admin key, never a session key:** code the policy runs may hold a session key, and must not
  grade itself.
- `columns` are prompt-side facts the product gives at the start (a surface, a task type, a cohort), on which credit
  may key; nothing the session does changes them.

**3.2 Admission (S1).** Sessions are admitted by serving capacity alone: under `serve.sessions.max_open` open
sessions, else 429 with `Retry-After`. Nothing about the trainer, its horizon or its staleness enters admission. This
departs from RFC 0005's "The gateway never returns 429" (its section 4) at session admission only. RFC 0005 can queue
every call because a harness spec turns the harness's retries off and gives it a timeout longer than the gateway's
queue; a product's client keeps its own retries and timeouts, so `platoon serve` sheds load where a session starts,
and the product's backend, which retries a 429 that carries `Retry-After`, backs off there. The calls of an admitted
session still queue, and fail as `error/gateway_overload` after the queue's timeout, as RFC 0005's section 4 designs.

**3.3 Clients platoon does not configure.** A product's client was not written for platoon, so `platoon serve`, and
only it, adds:
- **call kinds by evidence** (RFC 0005, section 3), with a rule for `main`, since a session carries no orchestrator
  instruction: the root's first call that no evidence marks opens its main path, and a call is `main` when its prompt
  extends a `main` call (`base_call_id`) or carries the summary of a compaction on that path; a minted agent's calls
  follow the same rule on its own path, under its role. RFC 0005's harness evidence, or a client's `aux` kind header,
  marks compaction, side and auxiliary calls. Any other call is `unknown` and never trains (one that model-written
  code makes with a leaked session key, say), where today an untagged call is a `policy` call and a client's title,
  summary and compaction calls would train;
- **refusals by value:** a `response_format` other than text, a non-zero penalty, a non-empty `logit_bias`,
  `store: true`, or any `previous_response_id` or `conversation` gets a 400 that names the field, where today these
  are ignored silently. No-op values pass, as `n` is refused today only when it is not 1
  (`platoon/gateway/dialects/openai_chat.py`), so a client that sends `store: false` on every Responses call, as Codex
  does, is served;
- **a record of what sampled:** each call's `SamplingParams`, kept out of datums;
- **disconnects and retries:** a call whose client disconnects runs to its end and is journaled with an annotation
  that its reply was not delivered, where today the disconnect cancels the call and nothing is recorded
  (`platoon/gateway/server.py`); an `Idempotency-Key` is RFC 0005's `retry_key`, so a retry returns the recorded reply
  instead of sampling again;
- **rendering in a process pool,** as RFC 0005's section 6 proposes, so that one long render does not stall other
  sessions' keepalives;
- **a required bind host and a configured public URL** for minted keys' `base_url`, where today a server binds every
  interface by default and a minted child's base URL is the request's origin.

**3.4 What it samples (S7).** `platoon serve` samples through a Tinker session of its own, which loads the sampler
path that the `sessions` trainer's activation record names (section 9), and each call records that checkpoint's
version. It never holds a `FollowingSamplingClient`, which borrows the trainer's own Tinker session and dies with it
(`platoon/train/workers/binding.py`).
- **Only a named save can serve.** A step that saves no full checkpoint publishes today through an ephemeral save
  (`Policy.save_for_sampler`, `platoon/train/policy/policy.py`), whose sampler has no path: an ephemeral sampler
  cannot be loaded by another Tinker session. The trainer therefore saves each sampler it publishes under a name, with
  a durable `tinker://` path (section 8).
- **Serving apart from the trainer rests on two properties.** Tinker session B must serve a sampler path that Tinker
  session A saved while A runs and steps on, which is UNVERIFIED; and it must go on serving it after A has ended: RFC
  0021's U10, which miles-tinker does not provide today (its release unloads the samplers a Tinker session saved) and
  which is UNVERIFIED on hosted Tinker. PO-0 tests both. If either fails, the options are a backend change, or
  exporting weights to the product's own inference engine (not built; RFC 0011), whose tokens would then need the
  ingest contract section 11 leaves out.
- **Full fine-tuning.** A full fine-tuning gateway such as miles-tinker's serves only the newest weights pushed to its
  engines (`platoon/train/workers/binding.py`), so there an older sampler path would not pin what serves, and a gated
  snapshot needs engines of its own. LoRA adapters come first.
- **Shadow mode** serves without training: a frozen Tinker policy (`serve.policy.load_checkpoint_path`), or the
  product's current model through RFC 0015's hosted pass-through (`serve.policy.hosted`), whose calls never train. A
  hosted shadow is a hosted `platoon serve` process as a whole, not a hosted policy inside a Tinker gateway.
- **One process first.** Shards (RFC 0005, section 6) wait for measured load.

### 4. The session journal (S2)

```text
<journal>/
  sources.json            each source's spec as platoon serve loaded it, which the trainer checks at load
  sessions/<session_id>/
    events.jsonl          L0 (RFC 0007): model_call, tool, agent, reward and annotation events, and the new ended
                          and resolved events
    requests.jsonl        a request record per call: the messages and tools new since the call it extends, filtered
    calls.tokens.zst      L3 (RFC 0007, from PO-2): per call, the prompt and sampled tokens, logprobs and version,
                          and its routing handle, when the sampler sent one
    session.json          L1: the envelope, a fold of L0 (source, columns, roles, how it ended, its outcome)
  resolved.jsonl          one line per resolution, in order (a session, or the sessions of one group), which the
                          trainer reads: ids, numbers and reason codes only, no content
  deleted.jsonl           tombstones: a session id and when; the directory is gone
```

- **A request record per call,** because a `ModelCallEvent` holds lengths, versions and times but no content (RFC
  0007, section 2), and a hosted call records only its reply (`platoon/gateway/hosted.py`). Without request records a
  shadow could measure no next state, and no held-out set could be drawn from traffic. A record holds the messages and
  tools as the client sent them, after the capture filters, and only what is new since the call its prompt extends
  (`base_call_id`, RFC 0007), so a session's records grow with its length, not its square.
- **Routing handles.** RFC 0007's sidecar keeps no routing handle, since routing data "stays in the backend"; a
  session's rows are rebuilt from the journal alone, hours or days after they were sampled, so the journal keeps each
  call's handle (section 8).
- **Durable before the reply.** A call's L0 line, request record and tokens are written and fsynced before its reply
  is sent, or, for a streamed reply, before the events that carry its id, so every receipt names a durable call; a
  call cut off by a crash before then never answered and never trains.
- **One writer.** `platoon serve` is the journal's only writer; judge services post verdicts through the feedback
  route like any reporter.
- **Deleting a session removes its directory** and appends a tombstone; the trainer skips a session whose directory
  is gone, and counts it. The session's line in `resolved.jsonl` stays, and holds nothing but ids, numbers and reason
  codes. Deletion removes the record, not what a checkpoint has already learned from it: a product that needs more
  sets its source's `hold_seconds` to at least its deletion window, so that no session trains before it.
- **ATIF is only an export** (RFC 0007's M4). The trainer reads L0 and L3, never ATIF and never a harness's own
  transcript.
- **Size.** RFC 0007's estimate of 1-2 MB per rollout with tokens puts 10,000 sessions a day at 10-20 GB a day (EST).
  Retention is the product's.

### 5. The resolver (S3, S4)

A **source** is a kind of session, named when the session starts, with its own resolution policy. The resolver runs
in `platoon serve`'s process, on the journal it writes.

```python
@dataclass(frozen=True)
class SourceSpec:                                   # serve.sources.<name>
    awaits: tuple[str, ...]                         # the reward names it waits for: ("accepted",), ("report", "judge")
    deadline_seconds: float                         # counted from the session's end
    reward: str                                     # a session reward, by name (RFC 0013)
    reward_options: Mapping[str, Any] = field(default_factory=dict)
    on_deadline: Literal["censor", "score_arrived"] | float = "censor"   # a float: the declared value of silence
    hold_seconds: float = 0                         # no session resolves earlier than this after its end
    group_by: str | None = None                     # a prompt-side column: regenerations, best-of-n, A/B samples
    group_size: int = 1                             # with group_by: the most sessions one group holds
    tokens: bool = True                             # the token sidecar for its calls (PO-2); false: shadowed only


@dataclass(frozen=True)
class SessionEvidence:                              # sessions.evidence/v1, additive: readers ignore unknown fields
    session_id: str
    source: str
    columns: Mapping[str, str]                      # prompt-side, given at the start
    calls: Sequence["CallSummary"]                  # call id, agent, role, kind, version, finish reason
    events: Sequence["RewardEvent"]                 # feedback and verdicts by at_call_id; a None value stays None
    annotations: Sequence["AnnotationEvent"]        # RFC 0007's: disconnects, hints, judges' notes
    ended: Literal["end", "idle", "max_seconds"]
    deadline_passed: bool


class SessionReward(Protocol):                      # pure and total, as RFC 0022's ProductionReward is
    def __call__(self, evidence: SessionEvidence, options: Any) -> "SessionScore": ...


@dataclass(frozen=True)
class SessionScore:
    reward: float | None                            # None: censored, and nothing of the session trains
    per_call: Mapping[str, float]                   # rewards per call, for estimators that need spans (section 7)
    components: Mapping[str, float]
    reason: str
```

- **Resolved** means the session has ended, `hold_seconds` have passed since, and every name in `awaits` has an
  event or the deadline after its end has passed. The reward then scores the evidence, and the resolver appends the
  session, its reward and the reason to `resolved.jsonl`. At the deadline, `censor` (the default) trains nothing,
  `score_arrived` scores what arrived, and a float is recorded, as an event the resolver reports, for each awaited
  name that has none: a source may declare, say, that a reply nobody rated before the deadline counts as rated 0.
- **Events apply once,** by `event_id`, so a retried report changes nothing. A reward reads every event a session
  holds, so which of two reports of one name counts is the reward's rule.
- **Judges are services.** A next-state judge, a hint writer or a classifier of follow-ups reads ended sessions and
  posts verdicts, so a judge outage delays resolution and never fails a session, which cannot be collected again. A
  judge may be a frozen Tinker policy or a hosted model (RFC 0022's M12b); a hosted judge writes text and scores no
  tokens. A policy that optimizes a frozen scorer finds its blind spots, so a judge of sessions needs the calibration
  RFC 0022 gives its audits (canaries with known verdicts, deep audits by a second model family) and is trusted only
  once PO-1 has measured it against human labels.
- **Groups only by a declared key.** With `group_by`, sessions that share the column's value form one group of at
  most `group_size`. The group closes once every member has resolved, or at the deadline counted from its first
  member's end; members still waiting then follow the source's `on_deadline`, and the trainer credits the members
  the group holds. Sessions are never grouped by arrival.
- **One reward shape, two users.** `SessionReward` and RFC 0022's `ProductionReward` are one shape over two kinds of
  evidence and two kinds of score. RFC 0022 registers its rewards with no component kind or entry-point group "until
  a second user needs one" (its section 9.3); `SessionReward` is that user, so PO-2 extracts `Reward[E, S]`, a pure
  and total function from evidence `E` to a score `S`, and the reward registry records the evidence schema each
  reward reads.
- **Censoring is measured before it is trusted.** Missing feedback correlates with outcomes (a satisfied user often
  leaves without a word), so the `sessions/*` metrics report resolution and censoring by outcome and by latency, and
  PO-1 measures them before any reward is designed on them.

### 6. The `sessions` trainer

**6.1 The shape.** RFC 0021's SFT trainer "never touches the dispatcher, a workflow, permits, the stale filter or the
horizon", and `sft` is built so: a cursor over a dataset as a checkpoint participant, atomic steps, updates that fail
closed (`platoon/train/trainer/sft.py`). The `sessions` trainer is the same with a journal in place of the dataset,
plus a stale check of its own on each row:

```python
class SessionsTrainer(TrainerBase):     # trainer: sessions; needs(): no rollouts, no evaluation, one trained policy
    def check(self, run):
        self.sources = load_sources(self.options.journal, self.options.sources)    # <journal>/sources.json
        self.cursor = _JournalCursor(self.options.journal, sources=self.sources)   # a checkpoint participant
        run.checkpoints.register(self.cursor)   # registered in check(), it resumes before any service opens

    async def ready(self, run):
        await super().ready(run)
        self.learner = run.learner(MAIN, objective=self.objective)    # trainer_options.loss_fn, cispo by default

    async def run_steps(self, run):
        while run.clock.step < run.plan.end_step and not run.shutdown.is_set():
            resolved = await run.wait(self.cursor.take(self.options.batch_size), name="journal", interruptible=True)
            groups = [_credit_group(resolution) for resolution in resolved]   # the converter, on the record
            for group in groups:
                await self.credit.assign(group)
                self.credit.settle(group)
            rows = _fresh_rows(groups, version=run.policies[MAIN].version, bound=self.options.max_staleness)
            ...                                 # submit and apply through the learner
            await run.end_step()                # a named sampler save, logged; a full save on cadence holds the cursor
```

- **Conversion is today's.** The trainer rebuilds each session's `TrajectoryCollection` and interactions from L0 and
  L3, as RFC 0007's `rebuild_datums` does, and converts them with `get_train_data_for_trajectory_collection`
  (`platoon/utils/tinker_data_processing.py`), a function of the record, into a `CreditGroup`, a plain dataclass that
  is already built outside workflows (`platoon/testing/credit.py`). The conversion settings are the trainer's options
  and are fingerprinted, so a resume that changes them stops.
- **Options and load checks.** Like `sft`, it takes as its own options the `train:` keys that only `rl` reads
  (`batch_size`, `max_staleness`, and `loss_fn`, whose default is `cispo`), which the registry refuses under any other
  trainer (`RL_ONLY_TRAIN_KEYS`, `platoon/train/config_defs.py`). At load it reads `<journal>/sources.json`, as `sft`
  reads its datasets' `dataset_info.json`, and refuses a source the journal does not declare.
- **Exact resume.** The cursor, the estimator's state and the policy's weights are saved at one point (RFC 0021's
  participant rules). After a crash the trainer reads again from the saved cursor: every session after it trains once
  into the resumed weights, nothing runs again, and nothing is lost. The samplers published since that save stay
  logged and serving; under `mode: always` the next publication steps serving back to the resumed weights, which
  `mode: evaluated` prevents.
- **Lulls.** `run.wait` heartbeats the watchdog every minute while the journal is reachable and empty, so a quiet
  night does not trip the 600 s watchdog (`watchdog.timeout_seconds`); a journal that cannot be read raises.
- **Bursts.** A step takes `batch_size` resolved sessions, so a burst only makes steps come sooner. A `sessions` run
  trains only sessions, so a burst crowds out nothing; a run that mixes sessions with rolled-out entries is out of
  scope (Alternatives).

**6.2 What it gives up.** Evaluation during training, which every trainer but `rl` refuses today
(`platoon/train/trainer/registry.py`), though section 9's gate evaluates with `platoon eval` anyway; rolled-out
entries in the same run, such as curated tasks that anchor against forgetting; and every other policy, frozen ones
included, and agent bindings, which trainers other than `rl` refuse (`policies:`, `agents:`). So per-tenant adapters
are one `sessions` run each, and a frozen teacher (section 7) or a per-role `kl` term of RFC 0022's `roles` waits for
PO-3. RFC 0022's case against a sibling trainer (it would port evaluation, admission, independent stepping and
multi-LoRA) does not bite: this trainer admits no rollouts and trains one policy.

### 7. Credit for single samples and per step

- **`running` by default.** A session is a group of one, so the trainer's default estimator is RFC 0022's `running`:
  decayed baselines per stratum of the prompt-side `columns`, shrunk toward parent strata, read at the version the
  session sampled (`TrainingSample.version`) and moved only in `settle`. Its statistics reach back as far as the
  trainer's staleness bound, which is long for sessions (section 8). As RFC 0022's M1 built it, `running` is chosen,
  checked and built per `environments[]` entry, and its check reads the entry's bound (`max_staleness`, else
  `train.max_staleness`), which a `sessions` config has neither of. PO-2 therefore builds it from
  `trainer_options.credit` and `credit_options`, with `CreditSetup.max_staleness` set to the trainer's `max_staleness`,
  and widens the check to read that bound, for `running` and for the roles' estimators under `roles`. For a source whose
  sessions resolve alone (no `group_by` in `sources.json`), the trainer refuses at load an estimator that does not set
  RFC 0022's `credits_single_members`, since centering within a group of one zeroes every advantage: AReaL's lesson.
- **Multi-agent sessions.** A product harness that mints a planner and executors credits them apart with RFC 0022's
  `roles`.
- **Per step (PO-3).** `TrainingSample.spans`, each call's range of target tokens in a merged datum, recorded where
  the converter merges, and `AgentRun.events`, the session's reward events by `at_call_id`, both behind a class flag
  `needs_spans`, so that rows keep today's bytes unless an estimator asks. RFC 0021 defers RFC 0001's reward events
  and per-call spans until "the first per-step credit user"; sessions judged per turn are it.
- **Distillation from hints** is a package estimator on those spans: per call, it scores `hint + prompt_t →
  completion_t` with a teacher of the student's tokenizer, and maps the scores onto the call's span. `reverse_kl`
  cannot: a teacher scores the datum's own prompt (`Policy.score`, `platoon/train/policy/policy.py`), and loading
  refuses a teacher of another tokenizer (`platoon/train/credit/reverse_kl.py`), so a hosted model can write hints but
  never score tokens. A hint is drawn only from what followed in the session.

### 8. Versions and staleness

- **Named, logged, never reused (S5).** At each step's end the trainer publishes the step's weights through a named
  sampler save (`save_weights_for_sampler`), or through a full save's sampler when one is due, and appends the version
  and the sampler's path to `publications.jsonl` in its run directory, fsynced before the activation hook may name
  it. It resumes above the highest version logged, so a row from a sampler published after the last full save counts
  at least one version old and never shares a number with a new sampler. RFC 0021's resume rule
  (`published_version + 1`, its section 11) counts only what a full save recorded: harmless where every surviving row
  predates the save, wrong for journaled sessions, which survive a crash whichever sampler they came from. Serving an
  `rl` run's samplers would need the same log; `rl` keeps today's rule.
- **What a named save costs.** Under LoRA, an adapter's size; under full fine-tuning, the model's, which the pruner
  bounds (section 9).
- **Per call.** Each call records the version of the checkpoint `platoon serve` loaded; a datum takes its oldest
  call's version, as today (`own_version` in `platoon/utils/tinker_data_processing.py`).
- **A bound that covers the wait.** The trainer counts its own versions, one per step of `batch_size` resolved
  sessions. A row ages about `(L + R) × λ / B + 2` versions (EST): L being the session's length, R its time to
  resolution (at most the larger of the deadline and the hold), λ the rate at which sessions resolve and B the batch,
  plus one for the batch it waits in and one for the activation lag. The bound covers the longest session and the
  deadline: a bound sized for typical waits would mask, by construction, the sessions whose feedback took longest.
  Section 12's settings (sessions of at most 2 h, a 2 h deadline, 64 sessions an hour, a batch of 32) give
  `(2 + 2) × 64 / 32 + 2 = 10`.
- **`expired`.** A resolved session whose every row is past the bound never trains; its outcome is `expired`, counted
  by source and latency. A partly stale session keeps its fresh rows (the per-datum filter), and
  `conversion.split_at_version_change` starts a new datum where a call's version changes, so one old turn does not
  mask a long session.
- **Routing replay.** A row carries its last call's routing handle, which the backend resolves at training time
  (`platoon/utils/tinker_data_processing.py`), and routing data "stays in the backend" (RFC 0007). Whether a backend
  keeps it for days is UNVERIFIED: RFC 0022's routing-replay check (its section 9.4) measures hours first, and a dense
  base avoids the question.
- **Losses.** Each row carries its sampler's logprobs, which `cispo`, the default, reads. The trainer refuses a loss
  that needs provenance kept on the service, such as `stable_reinforce` in newer Tinker releases, whose provenance
  "expires after inactivity" (tinker: `docs/api/trainingclient.md`): a session's rows train hours or days after they
  were sampled.

### 9. What serves (S7)

- **An activation record**, `activation.json` in the `sessions` run's directory, names the checkpoint `platoon serve`
  loads: its sampler path, its version and the gate's result. The step hook `activation` writes it, deciding for each
  publication (section 8) whether it serves; the trainer adds the hook with `mode: always` when the config lists
  none. Its `mode` is `always` (every published sampler serves at once; PO-2's only mode) or `evaluated`: a
  `platoon eval` run of the nominee on a frozen held-out set must not fall below the active checkpoint by more than a
  margin. The hook starts the evaluation and never awaits it, so a slow gate delays serving, never training.
- **The held-out set** is split off from traffic by user and time before anything trains, and its sessions never
  train and seed no training task. They reach `platoon eval` rebuilt as tasks through the seed source (section 10),
  or replayed and scored by their source's judges. PO-1 builds the first one.
- **The pruner pins what serves** (PO-2). It never deletes a sampler that `activation.json` names or the nominee a
  gate is evaluating, where today it keeps none for a sampling client "under a trainer that rolls nothing out
  (`trainer: sft`)" (`platoon/train/checkpoints/pruning.py`); with `checkpoint.keep_recovery` it also deletes the
  named sampler saves that nothing pins, so that under full fine-tuning they do not fill the service's disk.
- **Rollback** (PO-4) activates an older checkpoint. It needs that checkpoint's sampler path to serve still (U10),
  and the pruner to keep every sampler the record named within a rollback window, where today it assumes that "A
  checkpoint an earlier start saved serves no sampling client of this one" (`platoon/train/checkpoints/pruning.py`).
- **Per-tenant adapters.** One `sessions` run per tenant, each an adapter of one base on a shared service (RFC 0021's
  section 12, U1-U4, checked by RFC 0022's `check_shared_service`), so that one tenant's sessions never train
  another's weights: the cross-user contamination OpenClaw-RL's
  [issue 42](https://github.com/Gen-Verse/OpenClaw-RL/issues/42) raises.
- **Capture filters,** components the config names (`serve.journal.filters`), run before anything is written: they
  redact a request record (credentials, or personal data by the product's classifier) or refuse to record a session
  at all. A redaction reaches the tokens: the redacted call's tokens are dropped, and so are those of every later
  call whose prompt extends it, which hold the same tokens. Those calls stay as context in their records and never
  train, since a redacted prompt is not what was sampled; a product that wants such sessions to train keeps the
  secret out of the prompt before it reaches `platoon serve`. Consent, legal basis and retention stay the product's.
- **Monitors and stops.** Behaviour monitors over the trainer's steps and the served traffic (answer length, refusals,
  clarifying questions in place of actions, invalid tool calls, judge flags, feedback rates by cohort) raise alarms
  that stop the trainer through `run.shutdown` and freeze activation. They run on the alarms of RFC 0022's archive
  hook (its section 9.6), extracted when this second user arrives.

### 10. Logged sessions as environment-generator seeds

The cheapest step needs no gateway and no credit for single samples. An environment generator rebuilds a recorded
session and its workspace as a Harbor task; RFC 0022's production gates the task with controls, solves it K times
with the solver entry's route and audits it, and the K solves train as the solver's rows inside the production's
group. Reef's AutoEnvScaling recipe reports 100% valid and 24% faithful tasks with an agent author, against 30% and 6%
from a prompt designer (Reef: `recipes/beta/autoenvscaling/README.md`). It is also the only route by which sessions a
hosted model answered can teach anything, since hosted calls never train. Besides the toolkit's own code, it adds:
- **in the toolkit,** a `seeds build` input (RFC 0022, section 9.7) that reads recorded sessions (an export of RFC
  0007's records, or a harness's own logs), scans them for credentials and personal data and writes authoring tasks,
  and a judge spec for faithfulness; the toolkit's final check keeps its planted-secret check on every task (RFC
  0022's `materialize` has none);
- **in platoon** (EST 100-150 lines), a faithfulness audit, an inspection in which a judge reads the session and the
  task and says whether the task asks what the user needed, and the seed's origin in the evidence
  (`seed.origin: session`, an additive field of `envgen.evidence/v1`), so that tasks derived from sessions stay out
  of the held-out generated set and out of evaluation sets built from the same sessions, which they would
  contaminate.

It follows RFC 0022's second study.

### 11. What stays out of scope

- **The product's own:** its interface and its client's instrumentation, beyond a documented URL and header contract;
  consent, legal basis, retention periods and classifiers of personal data (platoon gives capture filters and
  deletion by session); routing users and splitting A/B traffic (platoon publishes versions and activation); what a
  signal means.
- **Tokens sampled outside platoon.** RFC 0005's ledger reuses only tokens platoon's gateway sampled, and renders
  again when the tail holds "an assistant message the gateway did not produce". Training on another engine's tokens
  needs an ingest contract of its own (prompt and completion ids, the sampler's logprobs, its identity, tokenizer and
  renderer fingerprints) and a correction for a behaviour policy platoon did not run, as YouTube's
  [off-policy REINFORCE](https://arxiv.org/abs/1812.02353) estimates one and caps its weights. Until then such traffic
  is context, like hosted calls.
- **Not now:** a critic (Tinker has no value head: RFC 0021's PPO scenario); evolving harness text (Reef's second
  surface); streaming tokens as they are sampled (the gateway's SSE is synthetic); masks for train-inference mismatch,
  until traffic comes from engines platoon does not control; a hosted arm beside a Tinker one (RFC 0022's open
  question 6); runs that mix sessions with rolled-out entries, and trainer-side pending groups (Alternatives).

### 12. Configuration and new names

```yaml
# platoon serve serve.yaml
serve:
  host: 127.0.0.1                         # required; the product's TLS proxy sits in front
  public_url: https://sessions.example.com    # the base_url of minted keys
  admin_key_env: PLATOON_SERVE_ADMIN_KEY  # the variable holding the key the product's backend uses
  policy:
    model_name: <base>
    renderer_name: <renderer>
    tinker_base_url: <service>
    activation: <the sessions run's directory>    # or load_checkpoint_path: <sampler path>, or hosted: {...}
  policies:                               # frozen arms, each at a fixed sampler path (section 3.1)
    control: {model_name: <base>, renderer_name: <renderer>, tinker_base_url: <service>,
              load_checkpoint_path: <sampler path>}
  agents: {roles: {control: {policy: control}}}   # a session whose root takes the role control samples it
  sessions: {max_open: 64, idle_timeout_seconds: 1800, max_seconds: 7200, roles: [worker, control], max_agents: 16}
  journal: {dir: <journal>, filters: [acme.capture:Redact]}
  sources:
    assistant:
      awaits: [accepted]
      deadline_seconds: 7200
      reward: accepted                    # a session reward a package registers
      on_deadline: censor
      tokens: true                        # the token sidecar (PO-2): sessions of this source train
```

```yaml
# platoon train sessions.yaml
trainer: sessions
train: {model_name: <base>, renderer_name: <renderer>, lora_rank: 32}
trainer_options:
  journal: <journal>
  sources: [assistant]
  batch_size: 32                          # resolved sessions per step
  max_staleness: 10                       # (2 h + 2 h) × 64 sessions an hour / 32 + 2 (section 8)
  credit: running
  credit_options: {strata: [surface, task_type], priors: <measured priors>}
  conversion: {split_at_version_change: true}
  hooks: [{hook: activation, options: {mode: evaluated, eval: <held-out eval config>, margin: 0.01}}]
checkpoint: {every: 10, recovery_every: 5}    # a crash trains at most 5 steps of sessions again
```

| Kind | Names (milestone) |
|---|---|
| commands and config | `platoon serve` and its `serve:` section: `serve.policy` (`activation`, `load_checkpoint_path`, `hosted`), `serve.policies` and `serve.agents` (frozen arms), `serve.sessions`, `serve.journal` and `serve.sources.<name>` (PO-1); `trainer: sessions` and its options (PO-2) |
| HTTP | the session routes of section 3 (PO-1) |
| Python | `SourceSpec`, `SessionEvidence`, `CallSummary`, `SessionReward`, `SessionScore` and `@session_reward`, and RFC 0001's `RewardEvent`, with `event_id` and `reporter` (PO-1); `SessionsTrainer`, its cursor and helpers private, and `Reward[E, S]`, extracted with RFC 0022's `ProductionReward` (PO-2); `TrainingSample.spans`, `AgentRun.events` and `needs_spans` (PO-3); the capture-filter protocol (PO-4) |
| components | the trainer `sessions` (PO-2), its name reserved at PO-0 as RFC 0021 reserved `interleaved`; the step hook `activation` (PO-2 with `mode: always`; PO-4 adds `evaluated` and rollback) |
| formats | the journal's layout with `sources.json`, request records, the `ended` and `resolved` events, `resolved.jsonl`, `deleted.jsonl` and `sessions.evidence/v1` (PO-1); `publications.jsonl` and `activation.json` (PO-2); `seed.origin` in `envgen.evidence/v1` (Seeds) |
| outcomes and metrics | `expired` (PO-2); `serve/*` (sessions, refusals, latency, render lag) and `sessions/*` (resolution latency, censoring by outcome, expired sessions, lag in versions) (PO-1, PO-2) |

### 13. Compatibility and other RFCs

**Compatibility.** `platoon serve` is a new command and `trainer: sessions` a new trainer, so a config that uses
neither writes today's bytes: no existing key, request field, record or metric changes, and the golden trace gates
every pull request. Nothing changes without a key: the refusals, call kinds by evidence and 429 at admission belong
to `platoon serve` alone, and rollouts keep today's behaviour (an untagged call is a `policy` call, no new refusal,
no 429). Named sampler saves and the publication log exist only in `sessions` runs.

| RFC | What this RFC does to it |
|---|---|
| [0001](0001-native-multi-agent.md) native multi-agent | **Builds** its `RewardEvent`, journaled with `at_call_id`, a new `event_id` and its `source` named `reporter`, and with PO-3 the per-call reward events its credit leaves for later. Its section 1 points here |
| [0005](0005-gateway-dialects-and-tito.md) gateway dialects | **Builds**, in `platoon serve`, its call kinds by evidence (section 3), with a rule for a session's `main` path, `retry_key` and rendering in a pool (section 6). **Departs** from "The gateway never returns 429" at session admission only; its section 4 points here |
| [0007](0007-records-and-atif.md) records | **Applies** its layout to sessions, one directory each, with request records and routing handles; builds its M1 and M3 (PO-1) and M2 (PO-2) for sessions; answers its open question 1 for sessions: tokens for every session of a source that trains |
| [0015](0015-hosted-models-and-judges-in-eval.md) hosted models | **Extends** its pass-through to `platoon serve`'s shadow mode, as a whole hosted process, and to judges of sessions; hosted calls still never train. It points here under its status |
| [0021](0021-trainers-credit-and-policies.md) trainers, credit and policies | **Adds** a trainer of its SFT trainer's shape; makes U10, and a Tinker session that serves another's saved sampler, conditions of S1 and S7, with a case in `check_shared_service`; publishes named samplers and never reuses a version number, which its resume rule allows for samplers published after the last full save. **Keeps** its deferral of a framework for producers of online auxiliary data: the `sessions` trainer is a trainer, not a framework |
| [0022](0022-programs-that-run-rollouts.md) programs that run rollouts | **Reuses**: `running` (M1, whose check PO-2 lets read the trainer's bound) and `roles` (M2) for credit, both built from the trainer's options; M3's rule for root roles, applied to session roots, for arms; M0's refusal, for S6 (M9's reserved roles need a program); frozen hosted policies (M12b) for hosted judges; `check_shared_service` (M7, to which PO-0 adds a case); the pure reward over evidence (M11), whose second user it is; and the alarms of its archive hook (M12). **Takes none** of its permits, drains or per-entry resume. **Adds** a seed source for its toolkit (section 10). The last row of its section 12, rollouts that platoon did not start, points here |
| [0013](0013-extension-model.md), [0016](0016-component-configuration.md), [0017](0017-public-surface-and-conformance.md), [0020](0020-rollout-permits.md) | RFC 0013: session rewards, capture filters, the hook and the trainer resolve by name. RFC 0016: their options sit next to their selectors. RFC 0017: every new name starts experimental. RFC 0020: sessions hold no rollout permits, and its refusal to let the horizon grow silently is one reason trainer-side pending groups are rejected |

## The hard parts

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| Serving cannot outlive the trainer's Tinker session: another Tinker session cannot load its named samplers, or U10 fails on the planned backend | high on miles-tinker today / high | PO-0's test before any serving code; S1 and S7 are conditional on it; else a backend change, or exported weights served elsewhere |
| Feedback is sparse, late and missing in ways that correlate with outcomes | high / high | censoring, never 0; resolution and censoring by outcome and latency, measured in PO-1 before any reward is designed |
| The policy games user signals: flattery, questions in place of work, tactics aimed at the users they work on | high / high | rewards gated by judges with audits; monitors that stop the trainer; the activation gate; per-tenant adapters; no reward pays for an invalid answer |
| Rows go stale before their evidence arrives | high / medium | a bound that covers the session, the deadline and the activation lag; `expired` counted; the per-datum filter and the split at version changes |
| Untrusted clients poison the data: a session key in code the policy runs, a hostile user, forged feedback | medium / high | S6; feedback with the admin key only; event ids; `unknown` calls never train; capture filters; per-tenant adapters |
| The journal grows and holds personal data | high / medium | a directory per session, deletion by directory, a hold before training, capture filters that drop the tokens of what they redact; retention is the product's; EST 1-2 MB a session with tokens |
| Named sampler saves fill the service under full fine-tuning, where a pinned path also cannot hold an older checkpoint | medium / medium | the pruner deletes those nothing pins; LoRA adapters first |
| Routing handles expire before days-old rows train on a mixture-of-experts base | unknown / high | RFC 0022's routing-replay check, extended to the deadline; a dense base until it passes |
| A shared service couples tenants: one tenant's load slows another's serving | medium / medium | one Tinker session per policy (RFC 0021); per-tenant metrics; separate services where it binds |
| A new process or trainer changes today's runs | low / high | both are new and selected by a key; the golden trace in every pull request |

## Alternatives

- **An inbox entry in `rl`**, `environments[].draws: inbox`, which turns each resolved session into a weighted draw,
  so that the horizon and the batch shares bound the sessions' share of a step beside other entries. It protects
  other entries from a burst only in a run that mixes them, since a step takes groups in queue order
  (`platoon/train/step/step.py`). Its price: draws that hold no permit and no start slot, members that only convert,
  a case in the dispatcher's readiness rule (`_entry_can_start`), a watchdog that knows lulls, an exemption from the
  stale-group watch (`StaleWatch`), an `expired` fate and a new dispatcher state format. The dispatcher, member and
  watchdog pieces alone come to EST 700-1,050 lines and 930 of tests, all in files that RFC 0022's permits, drains and
  stall watchdog change. **The decision rule:** the `sessions` trainer while product runs are separate, which is the
  default; the inbox only once a run that mixes sessions with rolled-out entries, such as a curated floor against
  forgetting, is a stated requirement. Its touches to RFC 0022's pieces (a draw without permits, a lull clause in the
  stall watchdog, resume per entry) are EST 5-20 lines each when it lands, after RFC 0022's M13.
- **Trainer-side pending groups**, RFC 0022's fallback (derived draws with pending parents): a group settles with its
  rows final and its rewards pending, and is credited when its evidence arrives. Feedback takes seconds to days or
  never comes, and holding it in the trainer costs what resolving in the journal avoids:
  - an `assign` that waits, which may take 600 s before it stops the run (`DEFAULT_ASSIGN_TIMEOUT_SECONDS`,
    `platoon/train/credit/api.py`);
  - a horizon slot held for days, which needs an exemption and a cap, where RFC 0020 forbids the horizon to grow
    silently;
  - bigger saves, since every full save writes waiting groups again at "about ten bytes per token"
    (`platoon/train/checkpoints/kept_groups.py`).

  The `sessions` trainer shares no code with RFC 0022's fallback. **What would change this:** rewards computed from
  rollouts the trainer runs, such as re-running a user's task K times in a sandbox to price a session, long enough
  that crash losses matter. Seeds from sessions do not: a seed is a durable task, and its production an ordinary
  episode tree.
- **Sessions as parked draws in the stock `rl` trainer** (EST 520 lines): a service whose sessions open as
  placeholder draws, a public front on the trainer's gateway, and feedback into the root's misc. It serves the product
  from the trainer's event loop, answers users with 429 while the horizon is full, cancels sessions at the rollout
  deadline and loses them in a crash, so S1 fails by construction, and its code is thrown away when `platoon serve`
  lands. It is acceptable only as a week-long probe of the reward and credit path on a developer's own sessions.
- **Serving from the trainer's sampler** (a `FollowingSamplingClient` in `platoon serve`). The freshest sampler, but
  it borrows the trainer's Tinker session: a fatal error there ends that Tinker session, the binding is published only
  under process isolation and only on the trainer's node, and serving dies with the trainer.
- **Serving only full saves.** No new kind of save, but every serving update then costs a full save of the weights and
  the optimizer's state, hundreds of gigabytes each for a large model trained in full
  (`platoon/train/checkpoints/pruning.py`), or serving moves only on the checkpoint cadence.
- **Training text rendered again** (OpenClaw-RL, Reef). Capture from any engine, cheaply, but it trains tokens the
  sampler never produced, with logprobs cut or padded to fit: S5 forbids it.
- **A database for the journal** (Trinity-RFT's SQL). Good for joins; files per session make deletion a directory
  removal, appends crash-safe and a bad session readable with `cat` and `jq`, as RFC 0007's own alternatives argue.
  An index over the files can come later.
- **Doing nothing.** Export product logs and fine-tune offline, which needs none of this but trains off-policy on
  behaviour days old; or stop at seeds (section 10), which trains on tasks rebuilt from sessions, never on the
  sessions themselves.

## Rollout plan

Seeds can land on their own after RFC 0022's second study. PO-0 comes before any product code, and PO-1 to PO-4
follow it in order, each in pull requests that keep the golden trace identical. Milestones are PO-0 to PO-4, apart
from RFC 0007's and RFC 0022's M-numbers, which this RFC cites. PO-1 reuses RFC 0022's rule for root roles (M3), its
judge-mint refusal (M0) and, for hosted judges, its hosted dispatch (M12b), so it follows them; PO-2 touches none of
the dispatcher, permit or per-entry resume code RFC 0022 changes. Lines are EST, code only.

| Milestone | Scope | Exit check | Pull requests |
|---|---|---|---|
| Seeds | section 10: in the toolkit, a session input for `seeds build` with its scans and a faithfulness judge spec; in platoon, the faithfulness audit, the seed's origin in the evidence and the exclusion from evaluation sets built from the same sessions (100-150) | sessions rebuilt as tasks pass controls and audits and train as the solver's rows of their productions; no held-out generated set, and no evaluation set built from the same sessions, holds one | |
| PO-0 decide | a pilot product or harness and its signals; a case in `check_shared_service` on the planned backend: Tinker session B serves a sampler path that Tinker session A saved under a name, while A steps on and after A has ended (U10); the trainer name `sessions` reserved | the reward source is real, and whether serving can outlive the trainer (S1, S7) | |
| PO-1 shadow | `platoon serve` (section 3), one process: sessions, minted keys, receipts, feedback, admission, the client hygiene; the journal's request records with RFC 0007's M1 and M3 events; the resolver, with nothing trained; a frozen Tinker policy, or the product's current model through the hosted pass-through (1,800-2,800) | feedback's rate and latency, and censoring by outcome; judges' agreement with human labels; a held-out set split off from real traffic by user and time | |
| PO-2 ingest | the `sessions` trainer (section 6); RFC 0007's M2 token sidecar; named sampler saves, `publications.jsonl` and versions never reused; `expired` and the split; `running` and `roles` built from `trainer_options`, their checks reading the trainer's bound; the activation record with `mode: always` and the pruner's pin; the product samples its own adapter of the same base, trained by a separate run; `Reward[E, S]` (900-1,450, plus the activation record and the extraction) | the loop trains from traffic; a sessions config with `credit: running` loads, and one without `max_staleness` is refused; a kill at every point between resolution, take, step, save and append trains each session once; a censored or expired session never trains; a resume that changes the conversion settings stops; no sampler that `activation.json` names is deleted | |
| PO-3 per step | spans and reward events behind `needs_spans`; a next-state judge service; distillation from hints as a package estimator; frozen scoring policies accepted under `trainer: sessions`, a registry change for it alone (200-300, plus package code) | credit per turn beats credit per session on the pilot's held-out set | |
| PO-4 governance | section 9: activation's `evaluated` mode, behind a `platoon eval` gate; rollback and its window, if U10 holds; per-tenant adapters; capture filters; monitors with stops on the alarms of RFC 0022's archive hook (piece 7's 300-500, less the `always` mode) | a planted regression never serves: the gate refuses it and the active checkpoint stays; a stop alarm halts the trainer and freezes activation; a planted secret appears in no journal file, its token ids included | |

## Open questions

1. **Is there a first product or harness, and whose data is it?** A developer's own Claude Code or Codex sessions need
   no consent machinery, but they train only through `platoon serve` with an open-weight policy, and training through
   Claude Code first needs RFC 0005's open question 1, on its terms of use, settled. Sessions a hosted model answered
   never train and can feed only the seeds. Other people's sessions need opt-in and deletion before PO-1. The
   maintainers decide, with the pilot's owner, at PO-0.
2. **Which feedback first?** Explicit reports (Reef, AReaL), judges of the next state (OpenClaw-RL), or sessions
   rebuilt as verifiable tasks (seeds). Proposed: seeds first, then reports, and judges of the next state only after
   PO-1 has measured them against human labels. Decide at PO-0.
3. **What should sessions sample?** Every sampler their own `sessions` run publishes (`mode: always`, proposed for
   PO-2), whose rows age in that run's versions; or a gated snapshot (`mode: evaluated`, Reef's choice, from PO-4),
   whose staleness is the release lag. On miles-tinker today both stop serving when the trainer's Tinker session
   ends (U10), so PO-0's test decides whether S1 is reachable without a backend change. Decide after that test.
4. **The `sessions` trainer or an inbox entry for PO-2?** Proposed: the `sessions` trainer, unless a run must anchor
   sessions with rolled-out curated tasks in the same steps. Decide before PO-2.
5. **429 at admission.** S1 departs from RFC 0005's section 4 for session admission. Proposed: yes, in `platoon serve`
   only; rollouts never get a 429. Decide before PO-1.
6. **What silence means.** Proposed: censoring by default; a source may declare a value for silence (0 for a reply
   nobody rated before the deadline, say), and no reward on a source is trusted before PO-1 has measured its
   censoring by outcome. Decide per source, in PO-1.
7. **Tokens for every session?** RFC 0007's open question 1, for sessions. Proposed: for every session of a source
   the trainer trains on (`tokens: true`), since a session cannot be collected again; sources that are only shadowed
   keep events and request records. Decide before PO-2.
8. **Is a hold enough for deletion?** Deleting a session removes its record, not what a checkpoint learned from it.
   Proposed: the hold (`hold_seconds` at least the product's deletion window); unlearning stays out of scope. Decide
   before PO-2.
