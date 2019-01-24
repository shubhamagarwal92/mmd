# Binarizes kb for the model
# Also append pad it to the model
# pad_id=0
# currently no end_id
# kb_vocab.pkl created here
import json
import argparse
import numpy as np
import pickle as pkl
import os
import nltk
from pprint import pprint
from collections import Counter

def read_vocab(data_dir):
	kb_vocab_stats_file = os.path.join(data_dir, "celeb_vocab_stats.pkl")
	kb_vocab_file = os.path.join(data_dir, "celeb_vocab.pkl")
	pad_id = 0 
	pad_symbol = '<pad>' 
	unk_id = 1 
	unk_symbol = '<unk>'
	start_id = 2 
	start_symbol = '<start>' 
	end_id = 3
	end_symbol='<end>'	
	cutoff = 1
	i = 4
	vocab_dict = {start_symbol:start_id, end_symbol:end_id,
						pad_symbol:pad_id, unk_symbol:unk_id}
	word_counter = pkl.load(open(kb_vocab_stats_file,'r'))
	# vocab_dict = {word:word_id for word_id, word in pkl.load(open(
	# 					vocab_file, "r"))[1].iteritems()}   
	vocab_count = [x for x in word_counter.most_common() if x[1]>=cutoff]
	for (word, count) in vocab_count:
		if not word in vocab_dict:
			vocab_dict[word] = i
			i += 1
	inverted_vocab_dict = {word_id:word for word, word_id in vocab_dict.iteritems()}
	both_dict = [vocab_dict, inverted_vocab_dict]
	with open(kb_vocab_file, 'wb') as f:
		pkl.dump(both_dict, f, protocol=pkl.HIGHEST_PROTOCOL)	
	return vocab_dict, inverted_vocab_dict

def pad_or_clip_utterance(utterance, max_len = 40, pad_id=0, end_id=3):
	length_utterance = len(utterance)
	if length_utterance>=(max_len):
		# utterance = utterance[:(max_len)]
		# For end id
		utterance = utterance[:(max_len-1)]
		utterance.append(end_id)
		seq_length = max_len
	else:
		# pad_length = max_len - length_utterance # pad = max - (length+end)
		# utterance = utterance + [pad_id]*pad_length
		# seq_length = length_utterance 
		pad_length = max_len - length_utterance -1 # pad = max - (length+end)
		utterance = utterance + [end_id] +[pad_id]*pad_length
		seq_length = length_utterance + 1
	return seq_length, utterance



def convert_celeb(data_dir, data_type, vocab_dict, pad_id=0, unk_id=2):
	kb_text_file_path = os.path.join(data_dir, data_type+"_celeb_data.txt")
	# kb_out_text_file_path = os.path.join(data_dir, data_type+"_kb_text.pkl")
	# kb_out_len_file_path = os.path.join(data_dir, data_type+"_kb_text_len.pkl")
	kb_out_file_path = os.path.join(data_dir, data_type+"_celeb_text_both.pkl")
	wpt = nltk.WordPunctTokenizer()

	kb_text_file = open(kb_text_file_path,'r')
	kb_utterance = []
	kb_seq_len = []
	for utterance in kb_text_file:
		utterance = utterance.strip()
		utterance_words = wpt.tokenize(utterance)
		utterance_word_ids = []
		for word in utterance_words:
			if word in vocab_dict:
				word_id = vocab_dict[word]
			elif word=='':
				word_id =pad_id #Corner case
			else:
				word_id =unk_id
			utterance_word_ids.append(word_id)
		length_utterance, utterance_word_ids = pad_or_clip_utterance(utterance_word_ids)
		kb_utterance.append(utterance_word_ids)
		kb_seq_len.append(length_utterance)
	# length_utterance, utterance_word_ids = pad_or_clip_utterance(utterance_word_ids)
	kb_corpora = [kb_seq_len,kb_utterance]

	with open(kb_out_file_path, 'wb') as f:
		pkl.dump(kb_corpora, f, protocol=pkl.HIGHEST_PROTOCOL)	


def main(args):
	vocab_dict, inverted_vocab_dict = read_vocab(args.data_dir)
	convert_celeb(args.data_dir, 'train', vocab_dict)
	convert_celeb(args.data_dir, 'valid', vocab_dict)
	convert_celeb(args.data_dir, 'test', vocab_dict)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)