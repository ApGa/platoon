FROM python:3.12-slim
RUN useradd --create-home --home-dir /srv/agent --uid 1000 agent && mkdir -p /app && chown agent:agent /app
WORKDIR /app
