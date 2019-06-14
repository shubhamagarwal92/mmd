import os
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import json
import cPickle as pkl
import random
import time
import math
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# plt.switch_backend('agg')
import numpy as np
import logging
import argparse
from annoy import AnnoyIndex
# Absolute imports
import mmd.utils.utils as utils
import mmd.modules.torch_utils as torch_utils
import mmd.modules.models as models

def check_cuda(seed):
	"""Check cuda"""
	if torch.cuda.is_available():
		use_cuda = True
		torch.cuda.manual_seed(seed)
	else:
		use_cuda = False
	return use_cuda

def print_model(model):
	for name, param in model.named_parameters():
		if param.requires_grad:
			print name
	print(model)

def main(args):
	config = utils.read_json_config(args.config_file_path)
	print(config)
	torch.manual_seed(config['training']['seed']) # Seed for reproducability
	use_cuda = check_cuda(config['training']['seed'])
	# Load vocabulary 
	with open(args.vocab_path, 'rb') as vocab_file:
		vocab = pkl.load(vocab_file)[1] #inverted_vocab
	vocab_size = len(vocab)
	# Server
	annoyIndex = AnnoyIndex(4096, metric='euclidean')
	annoyIndex.load(args.annoy_file_path)
	annoyPkl = pkl.load(open(args.annoy_pkl_path))
	# # Local
	# annoyIndex = ""
	# annoyPkl = ""
	# model_type = getattr(models, args.model_type)

	kb_len = None
	celeb_len = None
	kb_vec = None
	use_kb = False
	celeb_vec = None
	kb_size = None
	celeb_vec_size = None
	if args.use_kb=='True':
		use_kb = True
		celeb_data = pkl.load(open(args.test_celeb_path,'r'))
		kb_data = pkl.load(open(args.test_kb_path,'r'))
		kb_vocab = pkl.load(open(args.kb_vocab_path,'r'))
		celeb_vocab = pkl.load(open(args.celeb_vocab_path,'r'))
		kb_size = len(kb_vocab[0])
		celeb_vec_size = len(celeb_vocab[0])
		del kb_vocab, celeb_vocab

	if args.model_type == 'MultimodalHRED':
		model = MultimodalHRED(src_vocab_size=vocab_size,
							   tgt_vocab_size=vocab_size,
							   src_emb_dim=config['model']['src_emb_dim'],
							   tgt_emb_dim=config['model']['tgt_emb_dim'],
							   enc_hidden_size=config['model']['enc_hidden_size'],
							   dec_hidden_size=config['model']['dec_hidden_size'],
							   context_hidden_size=config['model']['context_hidden_size'],
							   batch_size=config['data']['batch_size'],
							   image_in_size=config['model']['image_in_size'],
							   bidirectional_enc=config['model']['bidirectional_enc'],
							   bidirectional_context=config['model']['bidirectional_context'],
							   num_enc_layers=config['model']['num_enc_layers'],
							   num_dec_layers=config['model']['num_dec_layers'],
							   num_context_layers=config['model']['num_context_layers'],
							   dropout_enc=config['model']['dropout_enc'],
							   dropout_dec=config['model']['dropout_dec'],
							   dropout_context=config['model']['dropout_context'],
							   max_decode_len=config['model']['max_decode_len'],
							   non_linearity=config['model']['non_linearity'],
							   enc_type=config['model']['enc_type'],
							   dec_type=config['model']['dec_type'],
							   context_type=config['model']['context_type'],
							   use_attention=config['model']['use_attention'],
							   decode_function=config['model']['decode_function'],
							   num_states=args.num_states,
							   use_kb=use_kb, kb_size=kb_size, celeb_vec_size=celeb_vec_size
							  )
	else:
		model = HRED(src_vocab_size=vocab_size,
				     tgt_vocab_size=vocab_size,
				     src_emb_dim=config['model']['src_emb_dim'],
				     tgt_emb_dim=config['model']['tgt_emb_dim'],
				     enc_hidden_size=config['model']['enc_hidden_size'],
				     dec_hidden_size=config['model']['dec_hidden_size'],
				     context_hidden_size=config['model']['context_hidden_size'],
				     batch_size=config['data']['batch_size'],
				     image_in_size=config['model']['image_in_size'],
		  		     bidirectional_enc=config['model']['bidirectional_enc'],
				     bidirectional_context=config['model']['bidirectional_context'],
				     num_enc_layers=config['model']['num_enc_layers'],
				     num_dec_layers=config['model']['num_dec_layers'],
				     num_context_layers=config['model']['num_context_layers'],
				     dropout_enc=config['model']['dropout_enc'],
				     dropout_dec=config['model']['dropout_dec'],
				     dropout_context=config['model']['dropout_context'],
				     max_decode_len=config['model']['max_decode_len'],
				     non_linearity=config['model']['non_linearity'],
				     enc_type=config['model']['enc_type'],
				     dec_type=config['model']['dec_type'],
				     context_type=config['model']['context_type'],
				     use_attention=config['model']['use_attention'],
				     decode_function=config['model']['decode_function'],
				     num_states=args.num_states,
				     use_kb=use_kb, kb_size=kb_size, celeb_vec_size=celeb_vec_size
				    )
	model = torch_utils.gpu_wrapper(model, use_cuda=use_cuda)
	# model = torch.load('model.pkl')
	model.load_state_dict(torch.load(args.checkpoint_path))
	model.eval()
	print_model(model)
	test_data = pkl.load(open(args.test_pkl_path,'r'))

	batch_size = config['data']['batch_size']
	total_samples = len(test_data)
	num_test_batch = int(math.ceil(float(total_samples)/float(batch_size)))
	sentences=[]
	# loss_criterion = nn.CrossEntropyLoss(ignore_index=config['data']['pad_id']) #weight=weight_mask) nn.CrossEntropyLoss
	# loss_criterion = torch_utils.gpu_wrapper(loss_criterion, use_cuda=use_cuda)
	for batch_id in range(num_test_batch):
		batch_start = time.time()
		batch_data = test_data[batch_id*batch_size:(batch_id+1)*batch_size]
		if use_kb:
			kb_len = np.array(kb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			kb_len = utils.convert_states_to_torch(kb_len, use_cuda=use_cuda)
			kb_vec = np.array(kb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			kb_vec = utils.convert_states_to_torch(kb_vec, use_cuda=use_cuda)
			# Celebs
			celeb_len = np.array(celeb_data[0][batch_id*batch_size:(batch_id+1)*batch_size])
			celeb_len = utils.convert_states_to_torch(celeb_len, use_cuda=use_cuda)
			celeb_vec = np.array(celeb_data[1][batch_id*batch_size:(batch_id+1)*batch_size])
			celeb_vec = utils.convert_states_to_torch(celeb_vec, use_cuda=use_cuda)

		text_enc_input, text_enc_in_len, image_enc_input, dec_text_input,\
			dec_out_seq, dec_seq_length= utils.get_batch_mmd_data(batch_data, config['data']['start_id'],
						config['data']['end_id'], config['data']['pad_id'],
						config['data']['image_rep_size'], annoyIndex, annoyPkl, 
						use_cuda=use_cuda, volatile=True)
		dec_output_prob = model(text_enc_input, image_enc_input, text_enc_in_len, 
						context_size=args.context_size,
						teacher_forcing_ratio=0, decode=True, use_cuda=use_cuda,
						kb_vec=kb_vec, celeb_vec=celeb_vec, kb_len=kb_len,
						celeb_len=celeb_len)
		dec_output_seq = dec_output_prob[:,0,:].data.cpu().numpy()
		# loss = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
		# 	dec_out_seq.view(-1))
		# dec_output_seq = dec_output_prob.data.cpu().numpy().argmax(axis=2) # argmax for each timestep
		for sequence in dec_output_seq:
			words = []
			for word_id in sequence:
				if word_id == config['data']['end_id']:
					break
				word = vocab[word_id]
				words.append(word)
			sentence = ' '.join(words)
			sentences.append(sentence)
	with open(args.out_file_path, 'w') as out_file:
		for item in sentences:
			out_file.write("{}\n".format(item))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-vocab_path', type=str, default='./data/vocab.pkl',
						help='path for vocabulary wrapper')
	parser.add_argument('-config_file_path', help='path to json config', required=True)
	parser.add_argument('-test_pkl_path', type=str, help='test_pkl_path')
	parser.add_argument('-checkpoint_path', type=str, help='checkpoint_path')
	parser.add_argument('-out_file_path', type=str, help='out_file_path')
	parser.add_argument('-annoy_file_path', type=str, help='annoy path')
	parser.add_argument('-annoy_pkl_path', type=str, help='annoy pkl')
	parser.add_argument('-model_type', type=str, default='MultimodalHRED', help='model type')
	parser.add_argument('-context_size', type=int, default=2, help='model type')
	parser.add_argument('-test_state_pkl_path', type=str, help='model type')
	parser.add_argument('-use_kb', type=str, default='False', help='whether to use kb')
	parser.add_argument('-test_kb_path', type=str, help='model type')	
	parser.add_argument('-test_celeb_path', type=str, help='celeb path')
	parser.add_argument('-celeb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-kb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-num_states', type=int, help='num states for multitasking')
	parser.add_argument('-out_class_file_path', type=str, help='out_file_path')
	args = parser.parse_args()
	main(args)
