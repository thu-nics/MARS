#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))

ROLL_PATH=${PWD}
export PYTHONPATH="$ROLL_PATH:$PYTHONPATH"

ROLL_OUTPUT_DIR="./runs/kuhn_poker/$(date +%Y%m%d-%H%M%S)"
ROLL_LOG_DIR=$ROLL_OUTPUT_DIR/logs
ROLL_RENDER_DIR=$ROLL_OUTPUT_DIR/render
export ROLL_OUTPUT_DIR=$ROLL_OUTPUT_DIR
export ROLL_LOG_DIR=$ROLL_LOG_DIR
export ROLL_RENDER_DIR=$ROLL_RENDER_DIR
mkdir -p $ROLL_LOG_DIR $ROLL_RENDER_DIR

python examples/start_agentic_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_kuhn_poker_gae | tee $ROLL_LOG_DIR/custom_logs.log