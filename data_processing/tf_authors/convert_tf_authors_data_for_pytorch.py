import os
import json
import pickle as pkl
import argparse

def find_first(item, vec):
	"""return the index of the first occurence of item in vec"""
	for i in xrange(len(vec)):
		if item == vec[i]:
			return i+1
	return -1

def save_to_pickle(obj, filename):
	# if os.path.isfile(filename):
	# 	self.logger.info("Overwriting %s." % filename)
	# else:
	# 	self.logger.info("Saving to %s." % filename)
	with open(filename, 'wb') as f:
		pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def convert_for_model(in_dir_path, out_dir_path, data_type, end_id, pad_id):
	data_path = os.path.join(in_dir_path, data_type+"_data_file.pkl")
	out_filename = os.path.join(out_dir_path, data_type+".pkl")
	data = pkl.load(open(data_path,'r')) 
	pickle_sentences = []
	for line in data:
		text_enc_input = line[0]
		text_enc_in_len = []
		for in_seq in text_enc_input:
			in_len = find_first(end_id, in_seq)
			# in_len = in_seq.index(end_id)
			if in_len==-1:
				text_enc_in_len.append(20)
			else:
				text_enc_in_len.append(in_len)
		image_input = line[1]
		image_input_len = [2, 2]
		dec_out_seq = line[2][1:] # no start token
		dec_out_seq.append(pad_id)
		dec_seq_length = find_first(end_id, dec_out_seq)
		# in_len = in_seq.index(end_id)
		if dec_seq_length==-1:
			dec_seq_length = 20
		pickle_sentences.append([text_enc_input, text_enc_in_len, image_input, image_input_len, dec_out_seq, dec_seq_length])
	# dec_seq_length = dec_out_seq.index(end_id)
	# dec_in_seq = dec_out_seq[:-1]
	# dec_in_seq.insert(0,start_id)
	# dec_in_seq = [pad_id if x==end_id else x for x in dec_in_seq]
	# dec_in_seq = [val.replace(end_id,pad_id) for val in dec_in_seq]
	# dec_in_seq = map(lambda x: pad_id if x==end_id, dec_out_seq)
		# pickle_sentences.append(line[0],line[1],line[2],line[3],line[4],line[4],dec_in_seq)
	save_to_pickle(pickle_sentences, out_filename)

if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-in_dir_path', type=str, help='in dir path')
	parser.add_argument('-out_dir_path', type=str, help='out dir path')
	args = parser.parse_args()
	convert_for_model(args.in_dir_path, args.out_dir_path, "train", 1, 3)
	convert_for_model(args.in_dir_path, args.out_dir_path, "valid", 1, 3)
	convert_for_model(args.in_dir_path, args.out_dir_path, "test", 1, 3)
