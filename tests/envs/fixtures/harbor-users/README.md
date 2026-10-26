# User fixtures

Harbor tasks written for platoon whose commands run as users other than the sandbox's own: they score 1 only where
each phase runs as the user its task names, as Docker runs them (`agent_user: honor`). The tests of
`tests/envs/test_harbor_users_singularity.py` run them on Singularity with `fakeroot: true`, and those of
`tests/integration/test_harbor_users_docker.py` on Docker. Each builds its image from `environment/Dockerfile`, so a
run fetches them as a dataset and builds their images first (`platoon prepare`); `image-user-65534` and
`workdir-after-user` defer a `USER`, which a fetch refuses unless its `refuse` sets `agent_user: false`.

| Task | What it checks |
|---|---|
| `agent-user` | `[agent].user = "agent"`: the agent works in a directory that the image gives it, as that user (uid 1000, its own `HOME`), cannot read the answer the image keeps to root, and appends to the file that the Dockerfile's last step copies in for it (`COPY --chown=agent:agent`, which the session applies) |
| `shared-root-verifier` | `[verifier].user = "root"` in Harbor's shared mode, with the agent as `agent`: the tests keep a secret as root and run the agent's script as nobody, which must not read it |
| `image-user-65534` | no `[agent].user`, and an image that ends with `USER 65534`, as nuclear-fusion-rl's do: the agent and the tests run as 65534, which alone may write the output directory that the image gives it |
| `workdir-after-user` | no users in `task.toml`, and a Dockerfile whose deferred `WORKDIR` follows `USER agent`, on `agent-user`'s build prefix: the agent and the tests run as `agent` in the working directory that the `WORKDIR` makes and gives it with the directory above it, as BuildKit does, while the directories the image has keep their owners and modes |
