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
read -p "Please enter context size: (as 2/5/10): " CONTEXT_SIZE
read -p "Please enter max len for data directory: (default as 20): " MAX_LEN

DATA_VERSION=${DATA_VERSION:-v2}
MAX_LEN=${MAX_LEN:-20}
CONTEXT_SIZE=${CONTEXT_SIZE:-2}

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
export DATA_DIR=$PARENT_DIR/data/
export OUT_PKL_DIR=$DATA_DIR/dialogue_data/context_${CONTEXT_SIZE}_${MAX_LEN}

echo "Creating kb vec in: "
echo $OUT_PKL_DIR

export VOCAB_PKL=$OUT_PKL_DIR/vocab.pkl


#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_kb_text.py \
-data_dir $OUT_PKL_DIR \

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_kb_input_pkl.py \
-data_dir $OUT_PKL_DIR \

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_vocab_with_kb.py \
-data_dir $OUT_PKL_DIR \
# -vocab_path $VOCAB_PKL



## OOM - do not use this
# srun --partition $QUEUE --nodelist=$NODE -c16 \
# python create_kb_vec.py \
# -data_dir $OUT_PKL_DIR \

