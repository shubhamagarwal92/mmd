import json
import argparse
import numpy as np
import pickle as pkl
import os
from collections import Counter
import nltk
from pprint import pprint

def convert_kb_vocab(data_dir, cutoff=2):
	kb_vocab_file = os.path.join(data_dir, "celeb_vocab_stats.pkl")
	original_vocab_file = os.path.join(data_dir, "vocab.pkl")
	new_vocab_file = os.path.join(data_dir, "vocab_with_celeb.pkl")

	word_counter = pkl.load(open(kb_vocab_file,'r'))
	original_vocab_dict = pkl.load(open(original_vocab_file,'r'))

	vocab_count = [x for x in word_counter.most_common() if x[1]>=cutoff]
	vocab_dict = original_vocab_dict[0]

	i = len(vocab_dict)
	print("Original vocab dict size {}".format(i))
	for (word, count) in vocab_count:
		if not word in vocab_dict:
			vocab_dict[word] = i
			i += 1
	inverted_vocab_dict = {word_id:word for word, word_id in vocab_dict.iteritems()}	
	both_dict = [vocab_dict, inverted_vocab_dict]
	with open(new_vocab_file, 'wb') as f:
		pkl.dump(both_dict, f, protocol=pkl.HIGHEST_PROTOCOL)

	print("New vocab dict size {}".format(len(vocab_dict)))

def main(args):
	convert_kb_vocab(args.data_dir)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)