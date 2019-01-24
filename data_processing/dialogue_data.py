# Adapted mostly from tf author's code
import os
import json
import copy
import cPickle as pkl
import argparse
import logging
import nltk
from nltk import word_tokenize
from itertools import izip
import collections
from collections import Counter
import parse_arguments as parse_arguments

class PrepareData():
	def __init__(self, context_size, max_images, user_start_id, user_end_id,
				 sys_start_id, sys_end_id, unk_id, pad_id, start_id, end_id, 
				 cutoff=-1):
		logging.basicConfig(level=logging.INFO)
		self.logger = logging.getLogger('prepare_data_for_hred')
		self.context_size = context_size
		self.max_images = max_images
		self.unk_id = unk_id
		self.pad_id = pad_id
		self.user_start_id = user_start_id
		self.user_end_id = user_end_id
		self.sys_start_id = sys_start_id
		self.sys_end_id = sys_end_id
		self.start_id = start_id
		self.end_id = end_id		
		self.user_start_sym = '<u>'
		self.user_end_sym = '</u>'
		self.sys_start_sym = '<s>'
		self.sys_end_sym = '</s>'
		self.pad_symbol = '<pad>'
		self.unk_symbol = '<unk>'
		self.start_sym = '<sos>'
		self.end_sym = '<eos>'	
		self.cutoff = cutoff # Vocab frequency cutoff
		self.dir_path = None
		self.data_type = None
		self.vocab_file = None
		self.vocab_dict = None
		self.word_counter = None
		self.context_text_file = None
		self.context_image_file = None
		self.user_state_file = None
		self.target_text_file = None
		self.wpt = nltk.WordPunctTokenizer()

	def save_to_pickle(self, obj, filename):
		if os.path.isfile(filename):
			self.logger.info("Overwriting %s." % filename)
		else:
			self.logger.info("Saving to %s." % filename)
		with open(filename, 'wb') as f:
			pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

	def prepare_data(self, dir_path, out_dir_path, vocab_file=None, vocab_stats_file=None, 
					data_type='train', create_vocab=False, write_all=True):
		""" write_all: False == not write nlg and images"""
		if not os.path.exists(out_dir_path):
			os.makedirs(out_dir_path)
		self.vocab_file = vocab_file
		self.vocab_stats_file = vocab_stats_file	
		self.data_type = data_type
		self.context_text_file = os.path.join(out_dir_path, self.data_type+"_context_text.txt")
		self.context_image_file = os.path.join(out_dir_path, self.data_type+"_context_image.txt")
		self.state_type_file = os.path.join(out_dir_path, self.data_type+"_state_type.txt")
		self.speaker_file = os.path.join(out_dir_path, self.data_type+"_speaker.txt")
		self.target_text_file = os.path.join(out_dir_path, self.data_type+"_target_text.txt")
		# KB related 
		self.sc_file = os.path.join(out_dir_path, self.data_type+"_search_criteria.txt")
		# self.sc_file = os.path.join(out_dir_path, self.data_type+"_actual_search_criteria.txt")
		self.cf_file = os.path.join(out_dir_path, self.data_type+"_compulsory_fields.txt")
		self.diff_file = os.path.join(out_dir_path, self.data_type+"_diff_fields.txt")
		self.name_file = os.path.join(out_dir_path, self.data_type+"_file_name.txt")
		# self.orientation_file = os.path.join(out_dir_path, self.data_type+"_orientation.txt")
		self.synset_file = os.path.join(out_dir_path, self.data_type+"_synset.txt")
		self.url_file = os.path.join(out_dir_path, self.data_type+"_url.txt")
		self.sc_pkl = os.path.join(out_dir_path, self.data_type+"_search_criteria.pkl")		
		self.actual_kb_file = os.path.join(out_dir_path, self.data_type+"_actual_kb.txt")
		self.kb_file = os.path.join(out_dir_path, self.data_type+"_kb.txt")
		# self.sc_val_pkl = os.path.join(out_dir_path, self.data_type+"_search_criteria_val.pkl")
		self.criteria_counter = Counter()
		# self.criteria_val_counter = Counter()

		self.read_dir_path = os.path.join(dir_path, self.data_type)
		self.read_jsondir(self.read_dir_path, create_vocab=create_vocab, write_all=write_all)
		if create_vocab:
			self.build_vocab(vocab_file, vocab_stats_file)
		self.save_to_pickle(self.criteria_counter, self.sc_pkl)
		# self.save_to_pickle(self.criteria_val_counter, self.sc_val_pkl)


	def read_jsondir(self, json_dir, create_vocab = False, write_all=True):
		if create_vocab:
			self.word_counter = Counter()
		else:
			self.word_counter = None
		for file in os.listdir(json_dir):
			if file.endswith('.json'):
				self.read_jsonfile(os.path.join(json_dir, file), create_vocab, write_all)

	def read_jsonfile(self, json_file, create_vocab, write_all):
		try:
			dialogue = json.load(open(json_file))
		except:
			return None
		filter(None, dialogue)
		context_text_list = []
		context_image_list = []
		target_text_list = []
		state_type_list = []
		dialogue_instance_multimodal = []
		dialogue_state_type_list=[]
		speaker_list = []
		index = 1
		kb_list = []
		sc_list=[]
		cf_list=[]
		diff_list=[]
		url_list=[]
		actual_kb_list = []
		filename_list = []
		synset_global_list = []
		global_kb_line = "" # Starting global kb line
		synset_line = ""
		sc_keys_line = ""
		for utterance in dialogue:
			actual_kb_line = "" # whenever the query is made
			url_line=""
			cf_line = ""
			diff_line = ""
			if utterance is None or len(utterance)==0:
				continue	
			if not isinstance(utterance, dict):
				print('impossible ', utterance, json_file)
				raise Exception('error in reading dialogue json')
				continue 
			speaker = utterance['speaker']
			if 'images' not in utterance['utterance'] or 'nlg' not in utterance['utterance']:
				continue
			images = utterance['utterance']['images']
			nlg = utterance['utterance']['nlg']
			if nlg is not None:
				nlg = nlg.strip().encode('ascii', 'ignore') #('utf-8')
			if nlg is None:
				nlg = ""
			nlg = nlg.lower().replace("|","")
			if create_vocab:
				try:
					nlg_words = self.wpt.tokenize(nlg)
					# nlg_words = nltk.word_tokenize(nlg)
				except:
					nlg_words = nlg.split(" ")
				self.word_counter.update(nlg_words)

			if 'search_criteria' in utterance['utterance']:
				kb_line = utterance['utterance']
				cf = kb_line['compulsory_fields']
				sc = kb_line['search_criteria']
				sc_keys = sc.keys()
				diff = list(set(sc_keys) - set(cf))
				cf_line = ';'.join(cf)
				diff_line = ';'.join(diff)
				sc_keys_line = ';'.join(sc_keys)
				local_kb_list = []
				local_kb_str = ""
				synset_list = []
				local_url_list = []
				for criteria in sc_keys:
					sub_criteria_keys = sc[criteria].keys()
					for sub_criteria in sub_criteria_keys:
						if criteria=='synsets':
							synset_list.append(sub_criteria)
						value = str(sc[criteria][sub_criteria])
						current_kb_str = '|'.join([criteria,sub_criteria,value])
						if criteria != 'url':
							counter_update_value = '|'.join([criteria,sub_criteria])
							self.criteria_counter.update([counter_update_value])
							local_kb_list.append(current_kb_str)
						else:
							local_url_list.append(current_kb_str)
				synset_line = ';'.join(synset_list)
				local_kb_str = ';'.join(local_kb_list)
				global_kb_line = local_kb_str  # Repeats KB line for next conversations
				actual_kb_line = local_kb_str # Only when KB query is there
				url_line = ';'.join(local_url_list)

			utt_type = utterance['type']
			utt_question_type = '-'
			utt_question_subtype = '-'
			if 'question-type' in utterance:
				utt_question_type = utterance['question-type']
			if 'question-subtype' in utterance:
				utt_question_subtype = utterance['question-subtype']
			state_string = speaker + ',' + utt_type + ',' + utt_question_type + ',' + utt_question_subtype
			dialogue_state_type_list.append(state_string) # Keeps track of states
			# dialogue_state_type_list.append({'speaker':speaker,'type':utt_type, 
			# 	'question_type':utt_question_type, 'utt_question_subtype':utt_question_subtype})
			dialogue_instance_multimodal.append({'images': images, 'nlg':nlg, 'speaker':speaker})				
			if speaker=="system":
				current_utterance = dialogue_instance_multimodal[-1]
				# Only when output is nlg 				
				if current_utterance['nlg'] is None or current_utterance['nlg']=="":
					continue
				padded_clipped_dialogue = self.pad_or_clip_dialogue(dialogue_instance_multimodal[:-1])
				# Checks
				if len(padded_clipped_dialogue)!=(self.context_size):
					raise Exception('some problem with dialogue instance, len != context_size')
				nlg_context =  [x['nlg'] if x['nlg'] is not None else '' for x in padded_clipped_dialogue]
				image_context = [x['images'] if x['images'] is not None else [] for x in padded_clipped_dialogue]
				speaker_context = [x['speaker'] if x['speaker'] is not None else '' for x in padded_clipped_dialogue]
				# Checks
				if len(nlg_context)!=self.context_size:
					raise Exception('len(nlg_context)!=self.context_size')
				if len(image_context)!=self.context_size:	
					raise Exception('len(image_context)!=self.context_size')
				target_text = dialogue_instance_multimodal[-1]['nlg']
				state_type = dialogue_state_type_list[-2] # -1 defines current state; -2 defines state before
				context_text_list.append(nlg_context)
				context_image_list.append(image_context)
				target_text_list.append(target_text)
				state_type_list.append(state_type)
				speaker_list.append(speaker_context)
				# KB related list 
				filename_list.append(json_file)
				actual_kb_list.append(actual_kb_line)
				kb_list.append(global_kb_line)
				cf_list.append(cf_line)
				diff_list.append(diff_line)
				synset_global_list.append(synset_line)
				url_list.append(url_line)
				sc_list.append(sc_keys_line)

		if write_all:
		# Writing to files
			with open(self.context_text_file, 'a+') as fp:
				for dialogue_instance in context_text_list:
					# dialogue_instance is a list
					dialogue_instance = '|'.join(dialogue_instance)
					fp.write(dialogue_instance+'\n')		
			with open(self.context_image_file, 'a+') as fp:
				for dialogue_instance in context_image_list:
					image_context = None
					if len(dialogue_instance)!=self.context_size:
						raise Exception('len(dialogue_instance_image_context_siz)!=self.context_size')		
					for images in dialogue_instance:	
						if image_context is None:
							image_context  = ",".join(images)
						else:	
							image_context = image_context+"|"+",".join(images)
					fp.write(image_context+'\n')
			with open(self.target_text_file, 'a+') as fp:
				for dialogue_instance in target_text_list:
					fp.write(dialogue_instance +'\n')
			with open(self.state_type_file, 'a+') as fp:
				for dialogue_instance in state_type_list:
					fp.write(str(dialogue_instance) +'\n')
			with open(self.speaker_file, 'a+') as fp:
				for speaker_instances in speaker_list:
					speaker_instance = ",".join(speaker_instances)
					fp.write(str(speaker_instance) +'\n')
		
		# New write operations
		self.write_list_to_file(self.url_file, url_list)
		self.write_list_to_file(self.kb_file, kb_list)
		self.write_list_to_file(self.actual_kb_file, actual_kb_list)
		self.write_list_to_file(self.sc_file, sc_list)
		# self.write_list_to_file(self.actual_sc_file, actual_sc_list)
		self.write_list_to_file(self.cf_file, cf_list)
		self.write_list_to_file(self.diff_file, diff_list)
		self.write_list_to_file(self.name_file, filename_list)
		# self.write_list_to_file(self.orientation_file, orientation_list)
		self.write_list_to_file(self.synset_file, synset_global_list)

	def write_list_to_file(self, out_file_path, out_list):
		""" Writes list to file"""
		with open(out_file_path, 'a+') as out_file:
		    for item in out_list:
		    	out_item = item.strip().encode('ascii', 'ignore')
		        out_file.write("{}\n".format(out_item))

	def pad_or_clip_dialogue(self, dialogue_instance):
		length_dialogue_instance = len(dialogue_instance)
		if length_dialogue_instance>=(self.context_size):
			return dialogue_instance[-(self.context_size):]
		else:
			pad_length = self.context_size - length_dialogue_instance
			padded_dialogue_instance = [{'images':[], 'nlg':'', 'speaker':''}]*pad_length
			padded_dialogue_instance.extend(dialogue_instance)
			return padded_dialogue_instance

	def build_vocab(self, vocab_file, vocab_stats_file):
		total_freq = sum(self.word_counter.values())
		self.logger.info("Total word frequency in dictionary %d ", total_freq)
		if self.cutoff != -1:
			self.logger.info("Cutoff %d", self.cutoff)
			vocab_count = [x for x in self.word_counter.most_common() if x[1]>=self.cutoff]
		else:
			vocab_count = [x for x in self.word_counter.most_common() if x[1]>=self.cutoff] # 5 cutoff frequency 
		self.vocab_dict = {self.start_sym:self.start_id, self.end_sym:self.end_id,
							self.pad_symbol:self.pad_id, self.unk_symbol:self.unk_id}
						# self.user_start_sym:self.user_start_id, self.user_end_sym:self.user_end_id,
						# 	self.sys_start_sym:self.sys_start_id, self.sys_end_sym:self.sys_end_id, 
		i = 4 # 6
		vocab_stats = []
		for (word, count) in vocab_count:
			if not word in self.vocab_dict:
				self.vocab_dict[word] = i
				i += 1
				vocab_stats.append([word, count, i])
		self.logger.info('Vocab size %d' % len(self.vocab_dict))
		self.save_to_pickle(vocab_stats, self.vocab_stats_file)
		inverted_vocab_dict = {word_id:word for word, word_id in self.vocab_dict.iteritems()}	
		both_dict = [self.vocab_dict, inverted_vocab_dict]
		self.save_to_pickle(both_dict, self.vocab_file)
		print('dumped vocab in ', self.vocab_file)

if __name__=="__main__":
	args = parse_arguments.arg_parse()
	preparedata = PrepareData(args.context_size, args.max_images, args.user_start_id, args.user_end_id, 
					args.sys_start_id, args.sys_end_id, args.unk_id, args.pad_id, args.start_id, 
					args.end_id, cutoff=args.cutoff)
	preparedata.prepare_data(args.dir_path, args.out_dir_path, args.vocab_pkl_path, 
							args.vocab_stats_path, data_type='train',create_vocab=True, write_all=True)
	preparedata.prepare_data(args.dir_path, args.out_dir_path, args.vocab_pkl_path, 
							args.vocab_stats_path, data_type='valid',create_vocab=False, write_all=True)
	preparedata.prepare_data(args.dir_path, args.out_dir_path, args.vocab_pkl_path, 
							args.vocab_stats_path, data_type='test',create_vocab=False, write_all=True)
