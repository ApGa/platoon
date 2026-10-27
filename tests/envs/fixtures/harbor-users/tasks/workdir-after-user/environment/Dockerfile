FROM python:3.12-slim
# agent-user's build prefix, whose image this task shares: the agent's user, and a directory only root may enter.
RUN useradd --create-home --home-dir /srv/agent --uid 1000 agent \
    && mkdir -p /app /srv/secret \
    && chown agent:agent /app \
    && echo 42 > /srv/secret/answer.txt \
    && chmod 700 /srv/secret && chmod 600 /srv/secret/answer.txt
USER agent
# Deferred: a directory the image has keeps its owner, and those a WORKDIR makes are its USER's, as BuildKit makes them.
WORKDIR /srv/secret
WORKDIR /srv/work/task
