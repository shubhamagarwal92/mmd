#!/usr/bin/env bash

python train.py \
-model_path $MODEL_DIR \
-vocab_path $VOCAB_PATH \
-train_pkl_path $TRAIN_PKL \
-valid_pkl_path $VALID_PKL \
-config_file_path $CONFIG_FILE_PATH \
-annoy_file_path $ANNOY_FILE \
-annoy_pkl_path $ANNOY_PKL \
-model_type $MODEL_TYPE \
-context_size $CONTEXT_SIZE \
-train_state_pkl_path $TRAIN_STATES_PKL \
-valid_state_pkl_path $VALID_STATES_PKL \
-train_kb_path $TRAIN_KB_PKL \
-valid_kb_path $VALID_KB_PKL \
-train_celeb_path $TRAIN_CELEB_PKL \
-valid_celeb_path $VALID_CELEB_PKL \
-kb_vocab_path $KB_VOCAB_PATH \
-celeb_vocab_path $CELEB_VOCAB_PATH \
-num_states $NUM_STATES \
-use_kb $USE_KB


