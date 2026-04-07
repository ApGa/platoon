# Email-Search Platoon Plugin

This plugin ports the ART-E email-search task into Platoon.

Citation: ported from [OpenPipe ART-E](https://github.com/OpenPipe/ART/tree/art-e/examples/art-e/art_e).

## Setup

Install the plugin:

```bash
cd plugins/email-search
uv sync --extra tinker
```

Generate the local email database once before running rollouts:

```bash
python -m platoon.email_search.data.local_email_db --overwrite
```

To keep the sqlite file off a slower repo-mounted drive, generate it in a custom location such as `/tmp`:

```bash
python -m platoon.email_search.data.local_email_db --db-path /tmp/enron_emails.db --overwrite
```

Then point `email-search` at that same file when running inference or training:

```bash
export PLATOON_EMAIL_SEARCH_DB_PATH=/tmp/enron_emails.db
```

## Training

Tinker:

```bash
python -m platoon.email_search.train_scripts.tinker.train_tinker
```

AReaL:

```bash
python platoon/email_search/train_scripts/areal/train_areal.py --config platoon/email_search/configs/areal/email_search_areal.yaml
```

## Inference

```bash
python -m platoon.email_search.inference_scripts.run_inference
```
