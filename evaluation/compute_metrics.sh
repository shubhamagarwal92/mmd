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
read -p "Please enter checkpoint epoch (like 20/30). Default 8: " CHECKPOINT_EPOCH

CHECKPOINT_EPOCH=${CHECKPOINT_EPOCH:-8}

export OUT_FILE_PATH=$RESULTS_DIR/pred_${CHECKPOINT_EPOCH}.txt
export TEST_TARGET_TOKENIZED=$RESULTS_DIR/test_tokenized.txt

srun --partition $QUEUE --nodelist=$NODE -c8 \
bash nlg_eval.sh > $RESULTS_DIR/metrics_tokenized_${CHECKPOINT_EPOCH}.txt
echo "Bleu done"
