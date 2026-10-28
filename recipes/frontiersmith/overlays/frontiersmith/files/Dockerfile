# main's image for FrontierSmith's tasks on platoon: Ubuntu 24.04 with the tools of the adapter's image (g++, Python 3
# with requests, curl, ripgrep), and none of its agents' installers (Claude Code, nvm and Codex, downloaded unpinned at
# build time): platoon brings its harnesses' own toolchains.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -o Acquire::Retries=5 \
    && apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        g++ python3 python3-pip curl ca-certificates ripgrep \
    && pip3 install --break-system-packages requests \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
