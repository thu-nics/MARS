RUN_PATH=/mnt/public/yuanhuining/repos/ROLL-final/ROLL/runs/tictactoe_single/20250910-163015
CKPT=checkpoint-180
OUTPUT_PATH=/mnt/public/yuanhuining/models/selfplay/tictactoe_single

mv ${RUN_PATH}/actor_train-1/${CKPT}/iter_0000001/mp_rank_01 ${RUN_PATH}/actor_train-0/${CKPT}/iter_0000001/
mv ${RUN_PATH}/actor_train-2/${CKPT}/iter_0000001/mp_rank_02 ${RUN_PATH}/actor_train-0/${CKPT}/iter_0000001/
mv ${RUN_PATH}/actor_train-3/${CKPT}/iter_0000001/mp_rank_03 ${RUN_PATH}/actor_train-0/${CKPT}/iter_0000001/

mkdir -p ${OUTPUT_PATH}
cp /mnt/public/yuanhuining/models/Qwen3-4B/config.json ${RUN_PATH}/actor_train-0/${CKPT}/
python mcore_adapter/tools/convert.py --checkpoint_path ${RUN_PATH}/actor_train-0/${CKPT}/ --output_path ${OUTPUT_PATH}

cp /mnt/public/yuanhuining/models/Qwen3-4B/generation_config.json ${OUTPUT_PATH}/