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
read -p "Please enter data version (as v1). Default v2: " DATA_VERSION
read -p "Please enter context size: (as 2/5/10): " CONTEXT_SIZE
read -p "Please enter max len for data directory: (default as 20): " MAX_LEN

# Default values
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
export CELEB_DIR=$DATA_DIR/data/meta_data/celebrity_profile/

export SYN_FILE=$CELEB_DIR/synset_distribution_over_celebrity_men.json
export CELEB_FILE=$CELEB_DIR/celebrity_distribution_over_synset_men.json

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python extract_celeb_profiles.py \
-data_dir $OUT_PKL_DIR \

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_celeb_vec.py \
-data_dir $OUT_PKL_DIR \
-syn_file $SYN_FILE \
-celeb_file $CELEB_FILE

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_vocab_with_celeb.py \
-data_dir $OUT_PKL_DIR \
# -vocab_path $VOCAB_PKL

#srun --partition $QUEUE --nodelist=$NODE -c16 \
python create_celeb_input_pkl.py \
-data_dir $OUT_PKL_DIR \
