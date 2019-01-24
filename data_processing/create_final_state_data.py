import os
import json
import cPickle as pkl
import argparse
import parse_arguments as parse_arguments

def create_state_data(out_dir_path, data_type):
	state_type_file = os.path.join(out_dir_path, data_type+"_state_type.txt")
	context_text_file = os.path.join(out_dir_path, data_type+"_context_text.txt")
	context_image_file = os.path.join(out_dir_path, data_type+"_context_image.txt")
	target_text_file = os.path.join(out_dir_path, data_type+"_target_text.txt")
	state_map_file = os.path.join(out_dir_path, "state_map.pkl") #data_type+"_state_map.pkl")
	index_map = pkl.load(open(state_map_file,'r')) # Index map
	with open(state_type_file, 'r') as state_file, \
		open(context_text_file, 'r') as text_file, \
		open(context_image_file, 'r') as image_file, \
		open(target_text_file, 'r') as target_file:
		for state_line, text_line, image_line, target_line in zip(state_file, text_file, \
			image_file, target_file):
			state = state_line.split(',')[2] # question type
			state_ = state.split('_')
			if state_[0]=='like' or state_[0]=='do':
				state = 'like'
			state_dir = str(index_map.index(state))
			state_out_dir = os.path.join(out_dir_path, state_dir)
			if not os.path.exists(state_out_dir):
				os.makedirs(state_out_dir)
			out_state_type_file = os.path.join(state_out_dir, data_type+"_state_type.txt")
			out_context_text_file = os.path.join(state_out_dir, data_type+"_context_text.txt")
			out_context_image_file = os.path.join(state_out_dir, data_type+"_context_image.txt")
			out_target_text_file = os.path.join(state_out_dir, data_type+"_target_text.txt")
			with open(out_target_text_file, 'a+') as fp:
				fp.write(target_line)
			with open(out_context_text_file, 'a+') as fp:
				fp.write(text_line)
			with open(out_context_image_file, 'a+') as fp:
				fp.write(image_line)
			with open(out_state_type_file, 'a+') as fp:
				fp.write(state_line)

if __name__=="__main__":
	args = parse_arguments.arg_parse()
	create_state_data(args.out_dir_path, "train")
	create_state_data(args.out_dir_path, "valid")
	create_state_data(args.out_dir_path, "test")