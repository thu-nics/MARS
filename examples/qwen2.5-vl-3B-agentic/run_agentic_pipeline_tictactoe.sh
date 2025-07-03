#!/bin/bash
set +x

CONFIG_PATH=$(basename $(dirname $0))
python examples/start_agentic_vl_pipeline.py --config_path $CONFIG_PATH  --config_name agentic_val_tictactoe | tee examples/logs/agent_val_tictactoe.log

