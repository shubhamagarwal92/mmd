export DATA_DIR=${PWD}/data/dataset/v2
export OUT_DIR=$DATA_DIR/analysis/valid # train/valid/test
mkdir -p $OUT_DIR
echo $OUT_DIR
export VALID_DIR=$DATA_DIR/valid
export OUT_FILE_JSON=$OUT_DIR/combined_df_all.json
export OUT_FILE_PKL=$OUT_DIR/combined_df_all.pkl

export OUT_FILE_JSON_USER=$OUT_DIR/combined_df_user.json
export OUT_FILE_PKL_USER=$OUT_DIR/combined_df_user.pkl

export OUT_FILE_JSON_SYS=$OUT_DIR/combined_df_sys.json
export OUT_FILE_PKL_SYS=$OUT_DIR/combined_df_sys.pkl

export STATS_FILE_JSON=$OUT_DIR/stats_df_all.json
export STATS_FILE_PKL=$OUT_DIR/stats_df_all.pkl

#srun --partition intel-longq --nodelist=dgx01 \
python combine_df.py -file_dir $VALID_DIR \
-output_file_json $OUT_FILE_JSON \
-output_file_pkl $OUT_FILE_PKL \
-output_user_file_json $OUT_FILE_JSON_USER \
-output_user_file_pkl $OUT_FILE_PKL_USER \
-output_sys_file_json $OUT_FILE_JSON_SYS \
-output_sys_file_pkl $OUT_FILE_PKL_SYS \
-stats_file_json $STATS_FILE_JSON \
-stats_file_pkl $STATS_FILE_PKL


