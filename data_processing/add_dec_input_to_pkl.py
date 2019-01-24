import os
import json
import pickle as pkl
import argparse
import parse_arguments as parse_arguments

def save_to_pickle(obj, filename):
	if os.path.isfile(filename):
		self.logger.info("Overwriting %s." % filename)
	else:
		self.logger.info("Saving to %s." % filename)
	with open(filename, 'wb') as f:
		pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


def convert_for_model(out_dir_path, data_type, seq_index, start_id, end_id, pad_id):
	data_path = os.path.join(out_dir_path, data_type+".pkl")
	out_filename = os.path.join(out_dir_path, data_type+"_.pkl")
	data = pkl.load(open(data_path,'r')) 
	pickle_sentences = []
	for line in data:
		dec_out_seq = line[seq_index]
		dec_in_seq = dec_out_seq[:-1]
		dec_in_seq.insert(0,start_id)
		dec_in_seq = [pad_id if x==end_id else x for x in dec_in_seq]
		# dec_in_seq = [val.replace(end_id,pad_id) for val in dec_in_seq]
		# dec_in_seq = map(lambda x: pad_id if x==end_id, dec_out_seq)
		pickle_sentences.append(line[0],line[1],line[2],line[3],line[4],line[4],dec_in_seq)
	save_to_pickle(pickle_sentences, out_filename)

if __name__=="__main__":
	args = parse_arguments.arg_parse()
	convert_for_model(args.out_dir_path, "train", 4, 2, 3, 0)
	convert_for_model(args.out_dir_path, "valid", 4, 2, 3, 0)
	convert_for_model(args.out_dir_path, "test", 4, 2, 3, 0)