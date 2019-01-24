import os
import json
import pickle as pkl
import argparse

def save_to_pickle(obj, filename):
	with open(filename, 'wb') as f:
		pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def main(args):
	inverted_vocab_dict = pkl.load(open(args.vocab_path,'r'))
	vocab_dict = {word:word_id for word_id, word in inverted_vocab_dict.iteritems()}	
	both_dict = [vocab_dict, inverted_vocab_dict]
	save_to_pickle(both_dict, args.out_vocab_path)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-vocab_path', type=str, help='in vocab path')
	parser.add_argument('-out_vocab_path', type=str, help='out vocab path')
	args = parser.parse_args()
	main(args)