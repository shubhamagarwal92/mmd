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

read -p "Please enter data version (as v2). Default v2: " DATA_VERSION
read -p "Please enter max len for data directory: (default as _20): " MAX_LEN
read -p "Please enter context size: (as 2/5/10): " CONTEXT_SIZE

DATA_VERSION=${DATA_VERSION:-v2}
MAX_LEN=${MAX_LEN:-20}
CONTEXT_SIZE=${CONTEXT_SIZE:-2}

export MAX_LEN=$MAX_LEN
export CONTEXT_SIZE=$CONTEXT_SIZE
export DATA_VERSION=$DATA_VERSION


export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
export DATA_DIR=$PARENT_DIR/data/dataset/$DATA_VERSION
export OUT_PKL_DIR=$DATA_DIR/dialogue_data/context_${CONTEXT_SIZE}_${MAX_LEN}

echo "Creating dataset in: "
echo $OUT_PKL_DIR

mkdir -p $OUT_PKL_DIR
export VOCAB_PKL=$OUT_PKL_DIR/vocab.pkl
export VOCAB_STATS=$OUT_PKL_DIR/vocab_stats.pkl

#srun --partition $QUEUE --nodelist=$NODE -c8 \
python dialogue_data.py \
-dir_path $DATA_DIR \
-out_dir_path $OUT_PKL_DIR \
-vocab_pkl_path $VOCAB_PKL \
-vocab_stats_path $VOCAB_STATS \
-max_len $MAX_LEN \
-context_size $CONTEXT_SIZE

#srun --partition $QUEUE --nodelist=$NODE -c8 \
bash create_final_state_data.sh

# export GLOBAL_OUT_PKL_DIR=$OUT_PKL_DIR/*/
# for folder in $GLOBAL_OUT_PKL_DIR; do
# export OUT_PKL_DIR=$folder
# srun --partition $QUEUE --nodelist=$NODE -c4 \
# python data_builder.py \
# -out_dir_path $OUT_PKL_DIR \
# -vocab_pkl_path $VOCAB_PKL \
# -max_len $MAX_LEN \
# -context_size $CONTEXT_SIZE
# done

# ### If we want to create a pkl for the whole dataset
# export OUT_PKL_DIR=$DATA_DIR/context_2
export OUT_PKL_DIR=$DATA_DIR/dialogue_data/context_${CONTEXT_SIZE}_${MAX_LEN}
#srun --partition $QUEUE --nodelist=$NODE -c4 \
python data_builder.py \
-out_dir_path $OUT_PKL_DIR \
-vocab_pkl_path $VOCAB_PKL \
-max_len $MAX_LEN \
-context_size $CONTEXT_SIZE
