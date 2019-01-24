# Write tokenized target to calculate bleu
import argparse
import logging
import nltk
import os

def main(args):
	wpt = nltk.WordPunctTokenizer()
	write_list = []
	with open(args.input_file, 'r') as in_file:
		for nlg in in_file:
			nlg_words = wpt.tokenize(nlg.strip())
			dialogue_instance = ' '.join(nlg_words)
			write_list.append(dialogue_instance)
	# Write list to file
	with open(args.tokenized_file, 'w') as out_file:
		for dialogue_instance in write_list:
			out_file.write(dialogue_instance+'\n')

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-input_file', type=str, help='path for input file')
	parser.add_argument('-tokenized_file', type=str, help='path for tokenized file')
	args = parser.parse_args()
	main(args)