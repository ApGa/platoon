```bash
cd plugins/issue-resolution-swesmith
source .venv/bin/activate
export TINKER_API_KEY=tml-dummy
export TINKER_BASE_URL=http://localhost:9000
export trial="skyrl-flame-swesmith-fft-cispo-$(date +%Y%m%d-%H%M%S)"
export TINKER_PREFIX_MISMATCH_DEBUG_DIR="/data/user_data/adityabs/platoon_skyrl_tinker/logs/swebench-platoon-tinker/${trial}/prefix_mismatch_debug"
export PLATOON_OPENHANDS_ACT_DEBUG_DIR="/data/user_data/adityabs/platoon_skyrl_tinker/logs/swebench-platoon-tinker/${trial}/openhands_utils_debug"
mkdir -p $TINKER_PREFIX_MISMATCH_DEBUG_DIR
mkdir -p $PLATOON_OPENHANDS_ACT_DEBUG_DIR
uv run --extra tinker python -m platoon.issue_resolution.train_tinker \
    --config platoon/issue_resolution/train_issue_resolution_fft_tinker.yaml \
    --tinker_base_url "$TINKER_BASE_URL" \
    --stats.trial_name "$trial"
```

pkill -f apptainer
pkill -f agent_server