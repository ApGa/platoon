FROM python:3.12-slim
# As nuclear-fusion-rl's images end: an output directory that only the image's user may write, and that user.
RUN mkdir -p /app/output && chown 65534:65534 /app/output && chmod 755 /app/output
WORKDIR /app
USER 65534
