import json
import argparse
import numpy as np
import pickle as pkl
import os
from collections import Counter
import nltk
from pprint import pprint


def extract_celeb(data_dir, data_type):
	state_type_file_path = os.path.join(data_dir, data_type+"_state_type.txt")
	context_text_file_path = os.path.join(data_dir, data_type+"_context_text.txt")
	celeb_out_file_path = os.path.join(data_dir, data_type+"_raw_celebs.txt")

	state_file = open(state_type_file_path,'r')
	text_file = open(context_text_file_path,'r')

	cel_list =[]
	for state_line, text_line in zip(state_file, text_file):
		state = state_line.split(',')[2] # 2nd state to check celebrity
		context_line = text_line.split('|')[-1] # last context
		cel_line = ""
		if state == 'celebrity':
			local_celeb_list = []
			pattern = 'cel_'
			context_words = context_line.strip().strip('?').split(' ')
			for word in context_words:
				if pattern in word:
					local_celeb_list.append(word)
			cel_line = ' '.join(local_celeb_list)
		cel_list.append(cel_line)

	with open(celeb_out_file_path, 'w') as fp:
		for instance in cel_list:
			fp.write(str(instance) +'\n')

def main(args):
	extract_celeb(args.data_dir, 'train')
	extract_celeb(args.data_dir, 'valid')
	extract_celeb(args.data_dir, 'test')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)