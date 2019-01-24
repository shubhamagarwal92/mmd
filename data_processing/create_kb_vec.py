import json
import argparse
import numpy as np
import pickle as pkl
import os
from collections import Counter
import nltk
from pprint import pprint

def convert_kb(data_dir, data_type):
	kb_path = os.path.join(data_dir, data_type+"_kb.txt")
	kb_vec_file = os.path.join(data_dir, data_type+"_kb_vec.pkl")
	kb_headers_file = os.path.join(data_dir, "kb_headers.pkl")
	# kb_vec_file = os.path.join(data_dir, data_type+"_kb_vec.txt")
	kb_file = open(kb_path,'r')
	kb_lines = kb_file.readlines()

	kb_headers = pkl.load(open(kb_headers_file,'r'))
	kb_vec_list = []
	for kb_line in kb_lines:
		empty_vec = [0]*len(kb_headers)
		kb_elements = kb_line.split(';')
		for instance in kb_elements:			
			kb_value = instance.split('|')
			if len(kb_value) != 1:
				criteria = kb_value[0]
				sub_criteria = kb_value[1]
				counter_value = ' '.join([criteria,sub_criteria])
				try:
					index = kb_headers[counter_value]
					empty_vec[index] = int(float(kb_value[2].strip()))
				except:
					print("Not found for {}".format(counter_value))
		kb_vec_list.append(empty_vec)

	with open(kb_vec_file, 'wb') as f:
		pkl.dump(kb_vec_list, f, protocol=pkl.HIGHEST_PROTOCOL)	


def main(args):
	convert_kb(args.data_dir, 'train')
	convert_kb(args.data_dir, 'valid')
	convert_kb(args.data_dir, 'test')




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)