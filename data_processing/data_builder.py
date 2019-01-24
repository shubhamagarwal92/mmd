# for server uncomment 13, 30 to 32, 178
import os
import cPickle as pkl
import argparse
import logging
from itertools import izip
import collections
from collections import Counter
import numpy as np
# from annoy import AnnoyIndex
# import h5py
import nltk
import parse_arguments as parse_arguments

class DataBuilder():	
	def __init__(self, context_size, max_images, unk_id, pad_id, end_id, image_rep_size, 
				max_len, annoy_file, annoy_pkl):
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger('Data Builder')
		self.image_rep_size = image_rep_size		
		self.max_images = max_images
		self.max_len = max_len
		self.unk_id = unk_id
		self.pad_id = pad_id
		self.end_id = end_id
		# self.pad_symbol = '<pad>'
		# self.unk_symbol = '<unk>'	
		self.context_size = context_size
		# self.annoyIndex = AnnoyIndex(4096, metric='euclidean')
		# self.annoyIndex.load(annoy_file)
		# self.annoyPkl = pkl.load(open(annoy_pkl))
		self.context_text_file = None
		self.context_image_file = None
		self.target_text_file = None
		self.vocab_dict = None
		self.wpt = nltk.WordPunctTokenizer()

	def write_model_data(self, out_dir_path, vocab_file, data_type):
		""" Actual wrapper function that is called """
		self.vocab_file = vocab_file
		self.data_type = data_type
		self.context_text_file = os.path.join(out_dir_path, self.data_type+"_context_text.txt")
		self.context_image_file = os.path.join(out_dir_path, self.data_type+"_context_image.txt")
		self.target_text_file = os.path.join(out_dir_path, self.data_type+"_target_text.txt")
		self.read_vocab(vocab_file)
		# self.h5_file_path = os.path.join(out_dir_path, self.data_type+".h5")
		# self.hf = h5py.File(self.h5_file_path, 'w')
		self.dialogue_pkl_file = os.path.join(out_dir_path, self.data_type+".pkl")
		self.convert_for_model(self.context_text_file, self.context_image_file, 
								self.target_text_file, self.data_type)	
		# self.hf.close()

	def convert_for_model(self, context_text_file, context_image_file, 
						target_text_file, data_type):
		""" Main function that does the pre-processing """
		binarized_corpus = []
		# binarized_text_context_seq = []
		# binarized_text_context_len = []
		# binarized_image_context_seq = []
		# binarized_image_context_len = []
		# binarized_target_seq=[]
		# binarized_target_len=[]
		unknowns = 0.
		num_instances = 0
		with open(context_text_file) as textlines, open(context_image_file) as imagelines,\
			open(target_text_file) as targetlines:
			for text_context, image_context, target in izip(textlines, imagelines, targetlines):
			# for text_context, target in izip(textlines, targetlines):
				binarized_text_context_seq = []
				binarized_text_context_len = []
				binarized_image_context_seq = []
				binarized_image_context_len = []
				binarized_target_seq=[]
				binarized_target_len=[]
				text_context = text_context.lower().strip()
				num_instances += 1
				if num_instances%10000==0:
					print 'finished ',num_instances, ' instances'
				# Text context
				utterances = text_context.split('|')
				#binarized_text_context = []
				text_context_seq = []
				text_context_len = []
				for utterance in utterances:					
					context_text, unknowns, context_length = self.text_processing(
																	utterance, unknowns)
					#binarized_text_context.append([context_text, context_length])
					text_context_seq.append(context_text)
					text_context_len.append(context_length)
				# binarized_text_context_seq.append(text_context_seq)
				# binarized_text_context_len.append(text_context_len)
				# Image contexts
				image_context = image_context.strip()
				image_turns = image_context.split('|')
				image_context_seq = []
				image_context_len = []
				# # @TODO Uncomment this
				for image_turn in image_turns:
					context_image, length_images = self.image_processing(image_turn)
					image_context_seq.append(context_image)
					image_context_len.append(length_images)
				# binarized_image_context_seq.append(image_context_seq)
				# binarized_image_context_len.append(image_context_len)

				# #################################
				# # Target
				# #binarized_target=[]
				# binarized_target_seq=[]
				# binarized_target_length=[]
				target_seq, unknowns, target_length = self.text_processing(
																target, unknowns)
				#binarized_target.append([utterance_word_ids, target_length])
				# binarized_target_seq.append(target_seq)
				# binarized_target_len.append(target_length)
				#binarized_corpus.append([binarized_text_context, binarized_image_context, binarized_target])
				binarized_corpus.append([text_context_seq, 
										text_context_len, 
										image_context_seq, 
										image_context_len, 
										target_seq, 
										target_length])
		#binarized_corpus = np.asarray(binarized_corpus)
		#self.hf.create_dataset(data_type, data=binarized_corpus)		
		self.save_to_pickle(binarized_corpus, self.dialogue_pkl_file)
		# self.save_to_h5py(binarized_text_context_seq, binarized_text_context_len, \
		# 	binarized_image_context_seq, binarized_image_context_len,\
		# 	binarized_target_seq, binarized_target_len)
		self.logger.info("Number of unknowns %d" % unknowns)
		# self.logger.info("Mean document length %f" % 
		# float(sum(map(len, binarized_text_context_len))/len(binarized_text_context_len)))
		# self.logger.info("Writing training %d dialogues (%d left out)" % 
		# (len(binarized_text_context_len), num_instances + 1 - len(binarized_text_context_len)))

	def save_to_pickle(self, obj, filename):
		if os.path.isfile(filename):
			self.logger.info("Overwriting %s." % filename)
		else:
			self.logger.info("Saving to %s." % filename)
		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

	def read_vocab(self, vocab_file):
		self.vocab_dict = {word:word_id for word_id, word in pkl.load(open(
							vocab_file, "r"))[1].iteritems()}   
		# print self.vocab_dict
		#vocab_file contains both (id2word,word2id). Thus, [1] 

	def text_processing(self, utterance, unknowns):
		utterance = utterance.strip()
		utterance_words = self.wpt.tokenize(utterance)
		utterance_word_ids = []
		for word in utterance_words:
			if word in self.vocab_dict:
				word_id = self.vocab_dict[word]
			elif word=='':
				word_id =self.pad_id #Corner case
			else:
				word_id =self.unk_id
				unknowns += 1
			utterance_word_ids.append(word_id)
		length_utterance, utterance_word_ids = self.pad_or_clip_utterance(utterance_word_ids)
		# utterance_word_ids is a list of max_len
		return utterance_word_ids, unknowns, length_utterance

	def pad_or_clip_utterance(self, utterance):
		# utterance: list of (<start> sent <end>)
		length_utterance = len(utterance)
		if length_utterance>=(self.max_len):
			# end_token = utterance[-1]
			utterance = utterance[:(self.max_len-1)]
			utterance.append(self.end_id)
			seq_length = self.max_len
		else:
			pad_length = self.max_len - length_utterance -1 # pad = max - (length+end)
			utterance = utterance + [self.end_id] +[self.pad_id]*pad_length
			seq_length = length_utterance + 1
		return seq_length, utterance

	def image_processing(self, image_turn):
		images = image_turn.split(",")
		images = self.pad_or_clip_images(images)
		len_images = 0
		image_ids = []
		for image in images:
			image_rep, len_images = self.get_image_representation(image, len_images)
			image_ids.append(image_rep)
		return image_ids, len_images

	def get_image_representation(self, image_filename, len_images):
		image_filename = image_filename.strip()	
		if image_filename=="":
			return image_filename, len_images
			# return [0.]*self.image_rep_size, len_images
		#FOR ANNOY BASED INDEX
		try:	
			len_images +=1
			# return [1.]*self.image_rep_size, len_images
			return image_filename, len_images
			# return self.annoyIndex.get_item_vector(self.annoyPkl[image_filename]), len_images
		except:
			return image_filename, len_images
			# return [0.]*self.image_rep_size, len_images

	def pad_or_clip_images(self, images):
		length_images = len(images)
		if length_images>=self.max_images:
			images = images[:self.max_images]
		else:
			pad_length = self.max_images - length_images
			padded_images = ['']*pad_length
			images.extend(padded_images)
			# padded_images.extend(images)	
			# images = images+['']*pad_length
		return images	

if __name__=="__main__":
	args = parse_arguments.arg_parse()
	data_builder = DataBuilder(args.context_size, args.max_images, args.unk_id, 
								args.pad_id, args.end_id, args.image_rep_size, 
								args.max_len, args.annoy_file, args.annoy_pkl)
	data_builder.write_model_data(args.out_dir_path, args.vocab_pkl_path, 
								  data_type='train')
	data_builder.write_model_data(args.out_dir_path, args.vocab_pkl_path, 
								  data_type='valid')
	data_builder.write_model_data(args.out_dir_path, args.vocab_pkl_path, 
								  data_type='test')	

