#!/bin/bash
set +x

ROLL_PATH="/workspace/ROLL-main"
CONFIG_PATH=$(basename $(dirname $0))
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"
python examples/start_agentic_rollout_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_rollout_sokoban

