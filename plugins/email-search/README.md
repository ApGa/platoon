# Email Search

This plugin ports the ART-E email-search task into Platoon.

Source: [OpenPipe ART-E](https://github.com/OpenPipe/ART/tree/art-e/examples/art-e/art_e).

## Install

```bash
cd plugins/email-search
uv sync --extra tinker
```

Generate the local email database before running rollouts:

```bash
uv run python -m platoon.email_search.data.local_email_db --overwrite
```

For faster local storage:

```bash
uv run python -m platoon.email_search.data.local_email_db \
  --db-path /tmp/enron_emails.db \
  --overwrite

export PLATOON_EMAIL_SEARCH_DB_PATH=/tmp/enron_emails.db
```

## Train

Tinker:

```bash
uv run python -m platoon.email_search.train_scripts.tinker.train_tinker
```

AReaL:

```bash
uv run python platoon/email_search/train_scripts/areal/train_areal.py \
  --config platoon/email_search/configs/areal/email_search_areal.yaml
```

## Inference

```bash
uv run python -m platoon.email_search.inference_scripts.run_inference
```
