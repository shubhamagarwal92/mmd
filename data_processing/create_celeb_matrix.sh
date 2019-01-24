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

export CURRENT_DIR=${PWD}
export PARENT_DIR="$(dirname "$CURRENT_DIR")"
export DATA_DIR=$PARENT_DIR/data/
export OUT_DIR=$DATA_DIR/dialogue_data/celeb/
mkdir -p $OUT_DIR

#srun --partition $QUEUE --nodelist=$NODE -c4 \
python create_celeb_matrix.py \
-data_dir $DATA_DIR \
-out_dir $OUT_DIR 
