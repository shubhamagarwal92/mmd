#!/usr/bin/env bash
#read -p "Enter queue: i:intel / a:amd  " Q
#case $Q in
#    a )
#		read -p "Enter gpu (write as 07/08):  " GPU
#		NODE=gpu$GPU
#		QUEUE=amd-longq
#		;;
#    i )
#		NODE=dgx01
#		QUEUE=intel-longq
#		;;
#esac

read -p "RESULTS_DIR: " RESULTS_DIR
read -p "Please enter checkpoint epoch (like 20/30). Default 7: " CHECKPOINT_EPOCH
read -p "Please enter context size. Default 2: " CONTEXT_SIZE
read -p "Please enter config version (as _v2). Default _v5: " CONFIG_VERSION



MAX_LEN=${MAX_LEN:-_20}
CONFIG_VERSION=${CONFIG_VERSION:-_v5}
CONTEXT_SIZE=${CONTEXT_SIZE:-2}
DATA_VERSION=${DATA_VERSION:-v2}

export MODEL_TYPE=MultimodalHRED

export PROJECT_DIR=/home/sagarwal/projects/mmd/
export DATA_DIR=$PROJECT_DIR/data/dataset/v2/dialogue_data/
export LABELS_FILE=$DATA_DIR/context_${CONTEXT_SIZE}_20/test_state_labels.txt
export HRED_MODEL_DIR=$PROJECT_DIR/models/
export MODEL_DIR=$HRED_MODEL_DIR/context_${CONTEXT_SIZE}_20/$MODEL_TYPE/$CONFIG_VERSION/
export RESULTS_DIR=$MODEL_DIR

CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-7}

export OUT_FILE_PATH=$RESULTS_DIR/pred_${CHECKPOINT_EPOCH}.txt
export TEST_TARGET_TOKENIZED=$RESULTS_DIR/test_tokenized.txt

export TEST_CONTEXT=$RESULTS_DIR/test_context_text.txt


#srun --partition $QUEUE --nodelist=$NODE -c8 \
python $PROJECT_DIR/utils/create_state_wise_pred.py \
-labels_file $LABELS_FILE \
-results_dir $RESULTS_DIR \
-pred $OUT_FILE_PATH \
-context $TEST_CONTEXT \
-target $TEST_TARGET_TOKENIZED \
-checkpoint $CHECKPOINT_EPOCH \
-beam $BEAM_SIZE 


# export GLOBAL_OUT_PKL_DIR=$OUT_PKL_DIR/*/
# export OUT_PKL_DIR=$folder
for folder in $RESULTS_DIR; do
export OUT_FILE_PATH=$folder/pred_${CHECKPOINT_EPOCH}.txt
export TEST_TARGET_TOKENIZED=$folder/test_tokenized.txt
srun --partition $QUEUE --nodelist=$NODE -c8 \
bash nlg_eval.sh > $folder/metrics_tokenized_${CHECKPOINT_EPOCH}.txt
done
