import json
import argparse
import numpy as np
import pickle as pkl
import os
from collections import Counter
import nltk
from pprint import pprint

def convert_kb(data_dir, data_type, create_headers=False):
	kb_path = os.path.join(data_dir, data_type+"_kb.txt")
	kb_text_file = os.path.join(data_dir, data_type+"_kb_text.txt")
	kb_vec_file = os.path.join(data_dir, data_type+"_kb_vec.pkl")
	kb_vocab_file = os.path.join(data_dir, "kb_vocab_stats.pkl")
	kb_headers_file = os.path.join(data_dir, "kb_headers.pkl")
	# kb_vec_file = os.path.join(data_dir, data_type+"_kb_vec.txt")
	criteria_counter = Counter()
	word_counter = Counter()
	wpt = nltk.WordPunctTokenizer()
	kb_file = open(kb_path,'r')
	kb_lines = kb_file.readlines()
	kb_list = []
	for kb_line in kb_lines:
		# name|casual-trousers|1.0;name|casual trousers|1.0;name|casual trouser|1.0
		out_line = ''
		kb_elements = kb_line.split(';')		
		instance_list = []
		for instance in kb_elements:			
			# instance_line =''
			kb_value = instance.split('|')
			if len(kb_value) != 1:
				criteria = kb_value[0]
				sub_criteria = kb_value[1]
				counter_update_value = ' '.join([criteria,sub_criteria]) 
				criteria_counter.update([counter_update_value])
				if criteria != 'taxonomy' and criteria != 'bestSellerRank' and criteria != 'reviewStars':
					instance_line = ' '.join([criteria,sub_criteria])
					word_counter.update(wpt.tokenize(criteria))
					word_counter.update(wpt.tokenize(sub_criteria))
					instance_list.append(instance_line)
		out_line = ' '.join(instance_list) 
		kb_list.append(out_line)

	with open(kb_text_file, 'w') as fp:
		for kb_instance in kb_list:
			fp.write(str(kb_instance) +'\n')
	# Only for training dataset
	if create_headers:
		criteria_keys = sorted(criteria_counter.keys())
		kb_headers = {k: v for v, k in enumerate(criteria_keys)}
		with open(kb_headers_file, 'wb') as f:
			pkl.dump(kb_headers, f, protocol=pkl.HIGHEST_PROTOCOL)	
		with open(kb_vocab_file, 'wb') as f:
			pkl.dump(word_counter, f, protocol=pkl.HIGHEST_PROTOCOL)

def main(args):
	convert_kb(args.data_dir, 'train', create_headers=True)
	convert_kb(args.data_dir, 'valid', create_headers=False)
	convert_kb(args.data_dir, 'test', create_headers=False)




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)