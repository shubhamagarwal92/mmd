export AUTHORS_DATA_DIR=/home/sagarwal/projects/mmd/data/dataset/v2/dump/
export OUT_DATA_DIR=/home/sagarwal/projects/mmd/data/dataset/v2/dialogue_data/authors/

python convert_tf_authors_data_for_pytorch.py \
-in_dir_path $AUTHORS_DATA_DIR \
-out_dir_path $OUT_DATA_DIR

cp $AUTHORS_DATA_DIR/vocab.pkl $OUT_DATA_DIR/vocab_authors.pkl

python convert_vocab_for_pytorch.py \
-vocab_path $OUT_DATA_DIR/vocab_authors.pkl \
-out_vocab_path $OUT_DATA_DIR/vocab.pkl