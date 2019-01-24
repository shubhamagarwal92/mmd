import os
import json
import cPickle as pkl
import argparse
import parse_arguments as parse_arguments
import numpy as np

def create_state_data(out_dir_path, data_type):
	state_type_file = os.path.join(out_dir_path, data_type+"_state_type.txt")
	state_map_file = os.path.join(out_dir_path, "state_map.pkl") #data_type+"_state_map.pkl")
	state_label_file = os.path.join(out_dir_path, data_type+"_state_labels.txt")
	state_label_pkl = os.path.join(out_dir_path, data_type+"_states.pkl")
	index_map = pkl.load(open(state_map_file,'r')) # Index map
	state_labels = []
	with open(state_type_file, 'r') as state_file:
		for state_line in state_file:
			state = state_line.split(',')[2] # question type
			state_ = state.split('_')
			if state_[0]=='like' or state_[0]=='do':
				state = 'like'			
			# state = ','.join(state_line.split(',')[:-1])
			state_label = int(index_map.index(state))
			state_labels.append(state_label)

	with open(state_label_file, 'w') as out_file:
	    for item in state_labels:
	        out_file.write("{}\n".format(str(item)))

	state_label_arr = np.asarray(state_labels)
	with open(state_label_pkl, 'w') as pkl_file:
		pkl.dump(state_label_arr, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-out_dir_path', type=str, help='out dir path')
	args = parser.parse_args()
	create_state_data(args.out_dir_path, "train")
	create_state_data(args.out_dir_path, "valid")
	create_state_data(args.out_dir_path, "test")