export DATA_DIR=${PWD}/data/dataset/v2
export PKL_DIR=$DATA_DIR/analysis/valid
mkdir -p $OUT_DIR
echo $OUT_DIR
export OUT_FILE_PKL=$PKL_DIR/combined_df_all.pkl
export OUT_FILE_PKL_USER=$PKL_DIR/combined_df_user.json #.pkl
export OUT_FILE_PKL_SYS=$PKL_DIR/combined_df_sys.json #.pkl
export STATS_FILE_PKL=$PKL_DIR/stats_df_all.pkl
export ANALYSIS_FILE=$PKL_DIR/analysis_v2_dev_sys.xlsx
# can also use export DATA_VERSION=$(basename $DATA_DIR)
export DATA_VERSION=${DATA_DIR##*/}
export DATA_TYPE=${PKL_DIR##*/}

#srun --partition intel-longq --nodelist=dgx01 \
python read_combine_df.py \
-user_file_pkl $OUT_FILE_PKL_USER \
-sys_file_pkl $OUT_FILE_PKL_SYS \
-stats_file_pkl $STATS_FILE_PKL \
-output_file_path $ANALYSIS_FILE

