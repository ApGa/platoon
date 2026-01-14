# platoon-openhands-rl

Platoon plugin for intermediate rewards with the OpenHands software agent SDK.

## Installation

This plugin depends on:
- **platoon** (core library)
- **platoon-openhands** (OpenHands plugin)
- **areal** backend (for RL training)

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Step-by-step installation

We recommend installing into a dedicated virtual environment (not in home directory for Babel space usage constraints). The instructions below use a custom location (`/data/user_data/<username>/uv_cache/platoon/`), but you can use any path. In Babel, use a compute node (not a CPU node) so that GPU is detected during torch installation

Assuming you are in project root directory.
```bash
# Create directory for the environment
export VIRTUAL_ENV=/data/user_data/<username>/uv_cache/platoon
export UV_CACHE_DIR=/data/user_data/<username>/uv_cache/.cache
uv sync --active --extra areal --extra wandb
mkdir -p /data/user_data/$USER/uv_cache/platoon
source /data/user_data/<username>/uv_cache/platoon/bin/activate
uv pip install -e plugins/openhands
uv pip install -e plugins/openhands_rl
```

### Verify installation

```bash
python -c "
from platoon.openhands import *
from platoon.openhands_rl import *
print('All packages imported successfully!')
"
```
