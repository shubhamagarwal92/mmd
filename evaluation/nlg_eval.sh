#!/usr/bin/env bash
# source deactivate
# source activate eval_nlg

# export OUT_FILE_PATH=/home/sagarwal/projects/mmd/models/context_2_20/MultimodalHRED/_v10/pred_10.txt
# export TEST_TARGET_TOKENIZED=/home/sagarwal/projects/mmd/models/context_2_20/MultimodalHRED/_v10/test_tokenized.txt

python nlg_eval.py \
-pred_file $OUT_FILE_PATH \
-ref_file $TEST_TARGET_TOKENIZED
