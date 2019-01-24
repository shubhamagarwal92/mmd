import os
import json
import cPickle as pkl
import argparse
import parse_arguments as parse_arguments

def create_state_map(out_dir_path): 
	# Defining state map
	state_type_file = os.path.join(out_dir_path, "train_state_type.txt")
	state_stats_file = os.path.join(out_dir_path, "state_stats.txt")
	state_map_file = os.path.join(out_dir_path, "state_map.pkl")
	state_map_txt = os.path.join(out_dir_path, "state_map.txt")
	state_file = open(state_type_file,'r')
	state_dic = {}
	for line in state_file:
		state = line.split(',')[2] # question type
		state_ = state.split('_')
		if state_[0]=='like' or state_[0]=='do':
			state = 'like'
		if state in state_dic:
			state_dic[state] +=1
		else:
			state_dic[state] = 1

	with open(state_stats_file, 'w') as stats_file:
		stats_file.write(json.dumps(state_dic)) # use `json.loads` to do the reverse

	map_state_index = state_dic.keys()
	with open(state_map_file, 'w') as map_file:
		pkl.dump(map_state_index, map_file, protocol=pkl.HIGHEST_PROTOCOL)

	with open(state_map_txt, 'w') as txt_file:
		txt_file.write("\n".join(map_state_index))



if __name__=="__main__":
	args = parse_arguments.arg_parse()
	create_state_map(args.out_dir_path) #, "train")