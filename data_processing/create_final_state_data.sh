#!/usr/bin/env bash

#srun --partition $QUEUE --nodelist=$NODE -c4 \
python create_final_state_map.py \
-out_dir_path $OUT_PKL_DIR

#srun --partition $QUEUE --nodelist=$NODE -c4 \
python create_final_state_data.py \
-out_dir_path $OUT_PKL_DIR

python create_final_state_labels.py \
-out_dir_path $OUT_PKL_DIR

export GLOBAL_OUT_PKL_DIR=$OUT_PKL_DIR/*/
for folder in $GLOBAL_OUT_PKL_DIR; do
export OUT_PKL_DIR=$folder
#srun --partition $QUEUE --nodelist=$NODE -c4 \
python data_builder.py \
-out_dir_path $OUT_PKL_DIR \
-vocab_pkl_path $VOCAB_PKL \
-max_len $MAX_LEN \
-context_size $CONTEXT_SIZE
done
