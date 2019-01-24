import json
import argparse
import numpy as np
import pickle as pkl
import os
from collections import Counter
import nltk
from pprint import pprint

def get_celeb_dist(celeb_data, celeb):
	# Endorsed by a celeb
	line = ""
	synset_dic = {}
	celebs = celeb_data.keys()
	if celeb in celebs:
		synsets = celeb_data[celeb].keys()
		for synset in synsets:
			attributes = celeb_data[celeb][synset].keys()
			for attribute in attributes:
				sub_attrs = celeb_data[celeb][synset][attribute].keys()
				for sub_attr in sub_attrs:
					value = celeb_data[celeb][synset][attribute][sub_attr]
					key = synset
					if key not in synset_dic:
						synset_dic[key] = value
					else:
						if synset_dic[key]<value:
							synset_dic[key] = value
		line = ' '.join(sorted(synset_dic, key=synset_dic.get))
	return line


def get_synset_dist(synset_data, synset):
	# Which celeb endorses product
	line = ""
	celeb_dic={}
	synsets = synset_data.keys()
	if synset in synsets:
		attributes = synset_data[synset].keys()
		for attribute in attributes:
			sub_attrs = synset_data[synset][attribute].keys()
			for sub_attr in sub_attrs:
				celebs = synset_data[synset][attribute][sub_attr].keys()
				for celeb in celebs:
					value = synset_data[synset][attribute][sub_attr][celeb]
				key = celeb
				if key not in celeb_dic:
					celeb_dic[key] = value
				else:
					if celeb_dic[key]<value:
						celeb_dic[key] = value
		line = ' '.join(sorted(celeb_dic, key=celeb_dic.get))
	return line


def get_celeb_vec(data_dir, data_type, syn_json_file, celeb_json_file):
	state_type_file_path = os.path.join(data_dir, data_type+"_state_type.txt")
	# context_text_file_path = os.path.join(data_dir, data_type+"_context_text.txt")
	celeb_file_path = os.path.join(data_dir, data_type+"_raw_celebs.txt")
	synset_file_path = os.path.join(data_dir, data_type+"_synset.txt")
	celeb_out_file_path = os.path.join(data_dir, data_type+"_celeb_data.txt")
	celeb_vocab_stats_file = os.path.join(data_dir, "celeb_vocab_stats.pkl")

	cel_dist_file = open(syn_json_file,'r') # Inverse nomenclature
	syn_dist_file = open(celeb_json_file,'r') # Inverse nomenclature
	state_file = open(state_type_file_path,'r')
	celeb_file = open(celeb_file_path,'r')
	synset_file = open(synset_file_path,'r')

	word_counter = Counter()
	wpt = nltk.WordPunctTokenizer()
	
	celeb_data = json.load(cel_dist_file)
	synset_data = json.load(syn_dist_file)

	cel_list =[]
	for state_line, raw_cel_line, synset_line in zip(state_file, celeb_file, synset_file):
		state = state_line.split(',')[2] # 2nd state to check celebrity
		raw_cel_line = raw_cel_line.strip()
		cel_line = ""
		if state == 'celebrity':
			# Check first if celebrity is there
			if raw_cel_line != "":
				celebs = raw_cel_line.strip().split(' ')
				local_syn_list = []
				for celeb in celebs:
					cel_local_line = get_celeb_dist(celeb_data, celeb)
					if cel_local_line != "":
						local_syn_list.append(cel_local_line)
				cel_line = ' '.join(local_syn_list)
			# Extract celebs endorsing this product					
			elif synset_line != "":
				synsets = synset_line.strip().split(';')
				local_celeb_list = []
				for synset in synsets:
					cel_local_line = get_synset_dist(synset_data, synset)
					if cel_local_line != "":
						local_celeb_list.append(cel_local_line)
				cel_line = ' '.join(local_celeb_list)

		word_counter.update(wpt.tokenize(cel_line))
		cel_list.append(cel_line)

	with open(celeb_out_file_path, 'w') as fp:
		for instance in cel_list:
			fp.write(str(instance) +'\n')

	with open(celeb_vocab_stats_file, 'wb') as f:
		pkl.dump(word_counter, f, protocol=pkl.HIGHEST_PROTOCOL)


def main(args):
	get_celeb_vec(args.data_dir, 'train', args.syn_file, args.celeb_file)
	get_celeb_vec(args.data_dir, 'valid', args.syn_file, args.celeb_file)
	get_celeb_vec(args.data_dir, 'test', args.syn_file, args.celeb_file)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-data_dir', type=str, default='./',
						help='data dir')
	parser.add_argument('-syn_file', type=str, default='./',
						help='data dir')
	parser.add_argument('-celeb_file', type=str, default='./',
						help='data dir')
	args = parser.parse_args()
	main(args)