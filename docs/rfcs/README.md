# Platoon RFCs

An RFC (request for comments) is a design document for a change that is too large or too consequential for a
single pull request: a new subsystem, a new or changed public contract, or work that spans several pull requests
or repositories. Writing the design down first lets reviewers agree on its shape before anyone writes the code,
and leaves a record of why platoon works the way it does.

For how platoon is built today, read the [architecture overview](../architecture.md). The everyday workflow
(setup, tests, commits and pull requests) is in [CONTRIBUTING.md](../../CONTRIBUTING.md).

## Index

| RFC | Title | Status |
|---|---|---|
| [0001](0001-native-multi-agent.md) | Native multi-agent and swarm support | Draft |
| [0002](0002-multi-cluster-placement.md) | Multi-cluster placement and elasticity | Draft |
| [0003](0003-multi-policy-training.md) | Multi-policy training across model sizes and architectures | Draft |
| [0004](0004-sandweave-integration.md) | sandweave integration | Draft |
| [0005](0005-gateway-dialects-and-tito.md) | Gateway dialects, TITO and the MCP tool plane | Draft |
| [0006](0006-backend-profiles-and-sdk-upgrade.md) | Backend profiles and the Tinker SDK upgrade | Draft |
| [0007](0007-records-and-atif.md) | Records and ATIF export | Draft |
| [0008](0008-config-and-cli.md) | Config and CLI redesign | Draft |
| [0009](0009-gateway-step-budgets.md) | Step budgets enforced at the gateway | Deferred |
| [0010](0010-harbor-registry-and-multi-container-tasks.md) | Harbor datasets, images, grading and multi-container tasks | Draft |
| [0011](0011-checkpoint-export-and-chat-template.md) | Checkpoint export with a matching chat template | Draft |
| [0012](0012-tool-call-parsing.md) | Tool-call parsing for any chat template | Draft |
| [0013](0013-extension-model.md) | The extension model | Draft |
| [0015](0015-hosted-models-and-judges-in-eval.md) | Hosted models and a judges switch for evaluation | Draft |
| [0016](0016-component-configuration.md) | Component configuration | Accepted |
| [0017](0017-public-surface-and-conformance.md) | Public surface and conformance | Draft |
| [0018](0018-configuration-only-projects.md) | Configuration-only projects | Draft |
| [0019](0019-training-mixtures.md) | Training mixtures: per-environment settings, task statistics and samplers | Draft |

`tests/test_rfc_index.py` checks that this table lists every RFC file with the title and status the RFC itself
gives, so add the row in the same pull request as the RFC.

## When to write an RFC

Write an RFC before you write the code when a change touches a public contract that code outside this
repository depends on (the agent API, `@pt.harness`, `@pt.program` and `EpisodeContext`, the environment service
protocol, the TOML harness spec format, the config schema, the `platoon` command line, the gateway's HTTP
endpoints or the record formats), changes what trains, adds a subsystem, a backend or a required dependency,
removes or renames public names, or needs several pull requests or repositories.
[When a change needs an RFC](../../CONTRIBUTING.md#when-a-change-needs-an-rfc) in CONTRIBUTING.md is the full
list, with what does not need one. If you are unsure, open an issue that describes the problem and ask.

## How to propose a change

1. **Start a discussion (optional).** For a large idea, open an issue first that states the problem and the
   rough direction, so you learn early whether the maintainers want it.
2. **Take a number.** Use the next free number: one more than the highest number in this directory and in the
   open pull requests that add RFCs. `0000` is the template. If two open RFCs collide, the one that merges
   second takes the next free number before it merges.
3. **Write it.** Copy the template and fill it in:

   ```bash
   cp docs/rfcs/0000-template.md docs/rfcs/0013-short-title.md
   ```

   The file name is the number and a few lower-case words joined by hyphens. Set `Status` to `Draft` and
   `Created` to today's date.
4. **Add it to the index.** Add a row to the table above with the status `Draft`.
5. **Open a pull request** that contains only the RFC and the index row. Pull request titles follow Conventional
   Commits ([Commits and pull requests](../../CONTRIBUTING.md#commits-and-pull-requests)), so title it like
   `docs: Add RFC 0013 on <topic>`.
6. **Review.** Discussion happens on the pull request. Answer comments by revising the text; questions that stay
   open go into the RFC's **Open questions**. A maintainer merges the RFC as a `Draft` once it is complete and
   coherent, so that issues, pull requests and other RFCs can link to it. Merging a draft does not accept it.
7. **Decide.** A maintainer accepts, rejects or defers the RFC in a pull request that changes its `Status` (and
   the index row). A rejected or deferred RFC keeps its text and gets a short **Decision** paragraph under its
   status table: why, and for a deferred one, what would bring it back.
8. **Implement.** Implementation pull requests link the RFC (see below). When the last milestone merges, a final
   pull request updates the RFC to match the code that landed and sets its status to `Implemented`.

## Lifecycle

```text
Draft ──> Accepted ──> Implemented
  │
  ├────> Rejected
  └────> Deferred ──> Draft (when it is picked up again)
```

| Status | Meaning | Who changes it |
|---|---|---|
| `Draft` | Proposed and open for comments. A draft may be merged so others can link it. | The author, when the RFC is first added |
| `Accepted` | The maintainers agree with the design; implementation can start, milestone by milestone. | A maintainer |
| `Implemented` | Every milestone has merged and the RFC describes the code that landed. | The author or a maintainer, in the last pull request |
| `Rejected` | The design will not be adopted. The **Decision** paragraph says why. | A maintainer |
| `Deferred` | Not now: the problem is real, but something else must happen first. The **Decision** paragraph says what. | A maintainer |

An accepted RFC can still change: if implementation shows that part of the design does not work, update the RFC
in the pull request that changes course, and say why in its description.

## Linking an RFC to its implementation

- **In each implementation pull request**, say which RFC and milestone it implements, with a link, for example
  `Implements RFC 0012, milestone M2`. A pull request that departs from the RFC updates the RFC in the same pull
  request, or explains why not.
- **In the RFC**, list the pull requests as they merge: in the `Implementation` row of the status table, and in
  the `Pull requests` column of the rollout plan.
- **When it is done**, the RFC is updated to the names and behavior that actually shipped, and its status
  becomes `Implemented`. From then on the code and its documentation are authoritative; the RFC records why.

## Format

Every RFC follows [the template](0000-template.md). It opens with a status table (`Status`, `Created`, `Authors`,
`Builds on`, `Discussion`, `Implementation`), then has these sections:

1. **Summary**: the proposal in a paragraph.
2. **Motivation**: what platoon cannot do today, and who needs it.
3. **Proposal**: the design, with typed sketches of new interfaces.
4. **Alternatives**: what else was considered, and why not.
5. **Rollout plan**: milestones, each with an exit check.
6. **Open questions**: what needs a decision, and by when.

An RFC may add a **The hard parts** section after the proposal for the problems it expects to be difficult.
Typed sketches show shape, not final names; the code that lands is authoritative. EST marks an estimate and
UNVERIFIED a claim nobody has checked yet. Citations of other projects name the project and a path inside it,
for example `tinker-cookbook: tinker_cookbook/renderers/base.py`.

RFCs 0001 to 0011 were written before the template. They follow the same sections, but they have no `Authors`,
`Discussion` or `Implementation` rows (0005 and 0008 list what has landed in an `Implemented in part` row
instead, and 0010 gained the rows when it was revised), and 0009 and 0011 give their status in a line under the
title.
