
read -p "Is train: (1 - Yes, 0 - no): " IS_TRAIN
read -p "Please enter data version. Default v2: " DATA_VERSION
read -p "Please enter context size. Default 2: " CONTEXT_SIZE
read -p "Please enter state data directory. Blank for all: " DATA_STATE_DIR
read -p "Please enter state model saved directory. Blank for all: " MODEL_STATE_DIR

# Model variant 
read -p "Enter model: m:multimodelHRED / h:hred / mt:multitask  " MODEL_INPUT
case $MODEL_INPUT in
    m ) 
		MODEL_TYPE=MultimodalHRED
		;;
    h ) 
		MODEL_TYPE=HRED
		;;
    mt ) 
		MODEL_TYPE=Multitask_MultimodalHRED
		;;

esac

read -p "Please enter config version (as _shahetal). Default _shahetal: " CONFIG_VERSION
read -p "Please enter checkpoint epoch (like 20/30). Default 10: " CHECKPOINT_EPOCH


MAX_LEN=${MAX_LEN:-_20}
CONFIG_VERSION=${CONFIG_VERSION:-_shahetal}
CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-10}
CONTEXT_SIZE=${CONTEXT_SIZE:-2}
DATA_VERSION=${DATA_VERSION:-v2}
NUM_STATES=${NUM_STATES:-10}

export NUM_STATES=$NUM_STATES
export MODEL_TYPE=$MODEL_TYPE
export DATA_STATE_DIR=$DATA_STATE_DIR
export MODEL_STATE_DIR=$MODEL_STATE_DIR
export CHECKPOINT_EPOCH=$CHECKPOINT_EPOCH
export MAX_LEN=$MAX_LEN
export CONFIG_VERSION=$CONFIG_VERSION
export CONTEXT_SIZE=$CONTEXT_SIZE
export DATA_VERSION=$DATA_VERSION

export PROJECT_DIR=${PWD}
export DATA_DIR=$PROJECT_DIR/data/dataset/${DATA_VERSION}/dialogue_data/
export CONTEXT_DATA_DIR=$DATA_DIR/authors/
export DIR_PKL=$CONTEXT_DATA_DIR/$DATA_STATE_DIR

# Common
export HRED_CODE_DIR=$PROJECT_DIR/mmd_code
export HRED_MODEL_DIR=$PROJECT_DIR/models/
export MODEL_DIR=$HRED_MODEL_DIR/authors/$MODEL_TYPE/$MODEL_STATE_DIR/$CONFIG_VERSION/

mkdir -p $MODEL_DIR

export CONFIG_FILE_PATH=$HRED_CODE_DIR/config_hred_mmd${CONFIG_VERSION}.json
export TRAIN_PKL=$DIR_PKL/train.pkl
export VALID_PKL=$DIR_PKL/valid.pkl
export TEST_PKL=$DIR_PKL/test.pkl
export VOCAB_PATH=$CONTEXT_DATA_DIR/vocab.pkl

export TRAIN_STATES_PKL=$CONTEXT_DATA_DIR/train_states.pkl
export VALID_STATES_PKL=$CONTEXT_DATA_DIR/valid_states.pkl
export TEST_STATES_PKL=$CONTEXT_DATA_DIR/test_states.pkl

export TRAIN_KB_PKL=$DIR_PKL/train_kb.pkl
export VALID_KB_PKL=$DIR_PKL/valid_kb.pkl
export TEST_KB_PKL=$DIR_PKL/test_kb.pkl

echo "Model saved in: "
echo $MODEL_DIR
echo "Following config: "
echo $CONFIG_FILE_PATH


export ANNOY_PATH=$PROJECT_DIR/data/image_annoy_index
export ANNOY_FILE=$ANNOY_PATH/annoy.ann
export ANNOY_PKL=$ANNOY_PATH/ImageUrlToIndex.pkl
# export ANNOY_PKL=$ANNOY_PATH/FileNameMapToIndex.pkl

export CHECKPOINT_PATH=$MODEL_DIR/model_params_${CHECKPOINT_EPOCH}.pkl
export RESULTS_DIR=$MODEL_DIR/$DATA_STATE_DIR
export OUT_FILE_PATH=$RESULTS_DIR/pred_${CHECKPOINT_EPOCH}.txt

export OUT_CLASS_FILE=$RESULTS_DIR/class_${CHECKPOINT_EPOCH}.txt

echo "Saving results to" 
echo $RESULTS_DIR
mkdir -p $RESULTS_DIR

if [ $IS_TRAIN == 1 ]; then
echo "Training"
#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c4 \
bash train.sh > $MODEL_DIR/logs.txt
fi
echo "Training done"

#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c4 \
bash translate.sh
echo "Translation done"

export TEST_TARGET=$DIR_PKL/test_target_text.txt
export TEST_CONTEXT=$DIR_PKL/test_context_text.txt
export LOG_BLEU_FILE=$RESULTS_DIR/bleu_${CHECKPOINT_EPOCH}.txt
export TEST_TARGET_TOKENIZED=$RESULTS_DIR/test_tokenized.txt
export LOG_BLEU_TOKENIZED=$RESULTS_DIR/bleu_tokenized_${CHECKPOINT_EPOCH}.txt

cp $TEST_CONTEXT $RESULTS_DIR
cp $CONFIG_FILE_PATH $RESULTS_DIR
export RESULTS_FILE=$RESULTS_DIR/results_${CHECKPOINT_EPOCH}.txt

#srun --partition $QUEUE --nodelist=$NODE --gres=gpu -c8 \
bash nlg_eval.sh > $RESULTS_DIR/metrics_tokenized_${CHECKPOINT_EPOCH}.txt
echo "Bleu done"
