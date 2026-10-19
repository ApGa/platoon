<!--
Title the pull request like a commit subject: a Conventional Commit with no scope and a summary in sentence
case without a final period, such as "fix: Close ORS sessions when a rollout is cancelled". CI checks it, and
the title becomes the commit subject on main when the pull request is squash-merged. CONTRIBUTING.md, under
"Commits and pull requests", lists the types.
-->

## Summary

<!-- What changes, in a few sentences. -->

## Why

<!-- The problem this solves, or what it makes possible. -->

## Links

<!-- "Fixes #123", or "Implements RFC 0012, milestone M2". Delete this section when there is nothing to link. -->

## Tests

<!-- The tests you added, and the commands you ran with what they showed. -->

## Checklist

- [ ] The title is a Conventional Commit, `<type>: <summary>`, with no scope and a sentence-case summary
      without a final period
- [ ] Every fix and feature has a test, and a fix's test fails without the fix
- [ ] `uvx pre-commit run --all-files` and the test suite pass
- [ ] `uvx pre-commit run ty --hook-stage manual --all-files` reports nothing new in the lines you changed
- [ ] The docs, examples and command-line help that the change affects are updated
- [ ] No secrets or machine-specific paths in code, configs or tests
- [ ] A change that needs an RFC (CONTRIBUTING.md, "When a change needs an RFC") links one, and a commit that
      breaks a public contract has `!` after its type and a `BREAKING CHANGE:` footer
