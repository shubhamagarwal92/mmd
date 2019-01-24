import os
import sys
sys.path.append('..')
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
import json
import cPickle as pkl
import random
import time
import math
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

def optimizer_wrapper(model, optim_type, lr, weight_decay):
	"""Wrapper for different optimizers"""
	# TODO model.parameters() as input instead of model
	if optim_type == 'adam':
		optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_type == 'adadelta':
		optimizer = optim.Adadelta(model.parameters(), lr=lr, weight_decay=weight_decay)
	elif optim_type == 'sgd':
		optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
	else:
		raise NotImplementedError("Not avaiable optimizer")
	# Initialize optimizers and criterion differently for encoder and decoder
	# encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
	# decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
	return optimizer

# Printing model paramters
def print_model(model):
	for name, param in model.named_parameters():
		if param.requires_grad:
			print name
	# params = list(model.parameters())
	# print(params)
	# https://discuss.pytorch.org/t/how-to-print-models-parameters-with-its-name-and-requires-grad-value/10778
	# for name, param in model.state_dict().items():
	# or 
	# for param in model.parameters():
	# 	print(param)
	### Printing model
	print(model)

def evaluate(valid_pkl_path, loss_criterion, model, config, vocab_size, annoyIndex, 
			annoyPkl, use_cuda=False, use_kb=False, valid_kb_path=None, 
			valid_celeb_path=None):
	model.eval()
	# with torch.no_grad():
	valid_data = pkl.load(open(valid_pkl_path,'r'))
	batch_size = config['data']['batch_size']
	total_samples = len(valid_data)
	num_valid_batch = int(math.ceil(float(total_samples)/float(batch_size)))
	total_loss =0.
	n_total_words = 0.
	correct = 0
	total = 0

	kb_len = None
	celeb_len = None
	kb_vec = None
	celeb_vec = None
	if use_kb:
		celeb_data = pkl.load(open(valid_celeb_path,'r'))
		kb_data = pkl.load(open(valid_kb_path,'r'))

	valid_start = time.time()
	for batch_id in range(num_valid_batch):
		batch_data = valid_data[batch_id*batch_size:(batch_id+1)*batch_size]
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
						dec_text_input, dec_out_seq, context_size=args.context_size, 
						teacher_forcing_ratio=1, decode=False,use_cuda=use_cuda,
						kb_vec=kb_vec, celeb_vec=celeb_vec, kb_len=kb_len, celeb_len=celeb_len)
		loss_val = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
			dec_out_seq.view(-1))
		n_words = dec_seq_length.float().sum().data[0]
		n_total_words += n_words
		# batch_loss = loss_val.data[0]/n_words	
		total_loss += loss_val.data[0]
		dec_out_model = dec_output_prob.data.cpu().numpy().argmax(axis=2) # argmax for each timestep
		# print(dec_out_model)
		# print(dec_out_model.shape)

		# Multi task learning
		# _, predicted = torch.max(outputs.data, 1)
		# total += labels.size(0)
		# correct += (predicted == labels).sum().item()

	# Printing epoch loss		
	# epoch_loss = total_loss / num_valid_batch		
	epoch_loss = total_loss / n_total_words
	valid_elapsed = (time.time() - valid_start)/60
	# if (epoch+1) % config['training']['log_every'] == 0:
	print('Valid Loss: Loss: %.6f, Perplexity: %5.4f, Run Time:%5.4f'
		  %(epoch_loss, np.exp(epoch_loss), valid_elapsed))
	print("")

def main(args):
	config = utils.read_json_config(args.config_file_path)
	torch.manual_seed(config['training']['seed']) # Seed for reproducability
	use_cuda = check_cuda(config['training']['seed'])
	logging.basicConfig(
		level=logging.INFO,
		format='%(asctime)s - %(levelname)s - %(message)s',
		# filename='log/%s' % (experiment_name),
		filemode='w')
	# define a new Handler to log to console as well
	console = logging.StreamHandler()
	# optional, set the logging level
	console.setLevel(logging.INFO)
	# set a format which is the same for console use
	formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
	# tell the handler to use this format
	console.setFormatter(formatter)
	# add the handler to the root logger
	# logging.getLogger('').addHandler(console)
	# print 'Reading data ...'
	print(config)
	vocab = pkl.load(open(args.vocab_path,'rb'))[1]
	vocab_size = len(vocab)
	# Server
	annoyIndex = AnnoyIndex(4096, metric='euclidean')
	annoyIndex.load(args.annoy_file_path)
	annoyPkl = pkl.load(open(args.annoy_pkl_path))
	# # Local
	# annoyIndex = ""
	# annoyPkl = ""
	model_type = getattr(models, args.model_type)

	kb_vec = None
	use_kb = False
	celeb_vec = None
	kb_size = None
	celeb_vec_size = None
	kb_len = None
	celeb_len = None
	if args.use_kb=='True':
		use_kb = True
		celeb_data = pkl.load(open(args.train_celeb_path,'r'))
		kb_data = pkl.load(open(args.train_kb_path,'r'))		
		## TODO - copy in all
		kb_vocab = pkl.load(open(args.kb_vocab_path,'r'))
		celeb_vocab = pkl.load(open(args.celeb_vocab_path,'r'))
		kb_size = len(kb_vocab[0])
		celeb_vec_size = len(celeb_vocab[0])
		del kb_vocab, celeb_vocab

	model = model_type(src_vocab_size=vocab_size, #config['model']['src_vocab_size'],
				tgt_vocab_size=vocab_size, #config['model']['tgt_vocab_size'],
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
	print_model(model)
	optimizer = optimizer_wrapper(model, config['training']['optimizer'], config['training']['lr'],
				config['training']['lr_decay'])
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
	losses = []
	# Weight masking for cross entropy loss; can also use ignore_index
	# weight_mask = torch.ones(config['model']['tgt_vocab_size'])
	# weight_mask = torch_utils.gpu_wrapper(weight_mask, use_cuda=False)
	# weight_mask[3] = 0 # weight_mask[tgt_dic['word2id']['<pad>']] = 0
	loss_criterion = nn.CrossEntropyLoss(ignore_index=config['data']['pad_id']) 
	#weight=weight_mask) nn.CrossEntropyLoss
	loss_criterion = torch_utils.gpu_wrapper(loss_criterion, use_cuda=use_cuda)
	# Load all training data
	train_data = pkl.load(open(args.train_pkl_path,'r'))
	# train_state_data = pkl.load(open(args.train_state_pkl_path,'r'))

	batch_size = config['data']['batch_size']
	total_samples = len(train_data)
	num_train_batch = int(math.ceil(float(total_samples)/float(batch_size)))
	for epoch in range(config['training']['num_epochs']):
		total_loss = 0. 
		n_total_words = 0.
		epoch_start = time.time()
		for batch_id in range(num_train_batch):
			batch_start = time.time()
			batch_data = train_data[batch_id*batch_size:(batch_id+1)*batch_size]
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
				# print(kb_len)
				# # kb_array = np.array(kb_array)
				# # kb_len = np.array(kb_array[0])
				# kb_vec = np.array(kb_array[1])				
				# kb_vec = utils.convert_states_to_torch(kb_vec, use_cuda=use_cuda)
				# celeb_vec = celeb_data[batch_id*batch_size:(batch_id+1)*batch_size]
			# batch_state = train_state_data[batch_id*batch_size:(batch_id+1)*batch_size]			
			text_enc_input, text_enc_in_len, image_enc_input, dec_text_input,\
				dec_out_seq, dec_seq_length= utils.get_batch_mmd_data(batch_data, 
						config['data']['start_id'],
						config['data']['end_id'], config['data']['pad_id'],
						config['data']['image_rep_size'], annoyIndex, annoyPkl,\
						use_cuda=use_cuda)
			# Forward + Backward + Optimize
			# model.zero_grad() ??? zero grad model or optim?
			# https://discuss.pytorch.org/t/do-i-need-to-do-optimizer-zero-grad-when-using-adam-solver/3235
			optimizer.zero_grad()
			dec_output_prob = model(text_enc_input, image_enc_input, text_enc_in_len, 
							dec_text_input, dec_out_seq, context_size=args.context_size, 
							teacher_forcing_ratio=1, use_cuda=use_cuda, kb_vec=kb_vec, 
							celeb_vec=celeb_vec, kb_len=kb_len, celeb_len=celeb_len)
			# loss = loss_criterion(dec_output_prob, dec_out_seq)
			loss = loss_criterion(dec_output_prob.contiguous().view(-1, vocab_size), #config['model']['tgt_vocab_size']),
				dec_out_seq.view(-1))

			# target_toks = dec_out_seq.ne(config['data']['pad_id']).long().sum().data[0]
			n_words = dec_seq_length.float().sum().data[0]
			n_total_words += n_words
			loss.backward()
			optimizer.step()
			# exp_lr_scheduler.step()
			# Gradient clipping to avoid exploding gradients
			nn.utils.clip_grad_norm(model.parameters(), config['training']['clip_grad'])
			batch_elapsed = (time.time() - batch_start)/60
			batch_loss = loss.data[0]/n_words # @TODO
			if (batch_id+1) % config['training']['log_every'] == 0:
				print('Batch Loss: Epoch [%d], Batch [%d], Loss: %.6f, Perplexity: %5.5f, Batch Time:%5.4f'
					  %(epoch+1, batch_id+1, batch_loss, np.exp(batch_loss), batch_elapsed)) 
			total_loss += loss.data[0]
			losses.append(batch_loss)
		epoch_loss = total_loss / n_total_words
		
		epoch_elapsed = time.time() - epoch_start
		# if (epoch+1) % config['training']['log_every'] == 0:
		print('Epoch Loss: Epoch [%d], Loss: %.6f, Perplexity: %5.5f, Epoch Time:%5.4f'
			  %(epoch+1, epoch_loss, np.exp(epoch_loss), epoch_elapsed)) 
		if (epoch+1) % config['training']['evaluate_every'] == 0:
			print("\nEvaluation:")
			evaluate(args.valid_pkl_path, loss_criterion, model, config, vocab_size,\
					annoyIndex, annoyPkl, use_cuda=use_cuda, use_kb=use_kb, 
					valid_kb_path=args.valid_kb_path, valid_celeb_path=args.valid_celeb_path)
		# Save the models
		if (epoch+1) % config['training']['save_every'] == 0:
			# Save and load only the model parameters(recommended).
			torch.save(model.state_dict(), os.path.join(args.model_path, 
									'model_params_%d.pkl' %(epoch+1)))

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-model_path', type=str, default='./models/',
						help='path for saving trained models')
	parser.add_argument('-vocab_path', type=str, default='./data/vocab.pkl',
						help='path for vocabulary wrapper')
	parser.add_argument('-config_file_path', help='path to json config', required=True)
	parser.add_argument('-train_pkl_path', type=str, help='data directory path')
	parser.add_argument('-valid_pkl_path', type=str, help='data directory path')
	parser.add_argument('-annoy_file_path', type=str, help='annoy path')
	parser.add_argument('-annoy_pkl_path', type=str, help='annoy pkl')
	parser.add_argument('-model_type', type=str, default='MultimodalHRED', help='model type')
	parser.add_argument('-context_size', type=int, default=2, help='model type')
	parser.add_argument('-train_state_pkl_path', type=str, help='model type')
	parser.add_argument('-valid_state_pkl_path', type=str, help='model type')
	parser.add_argument('-use_kb', type=str, default='False', help='whether to use kb')
	parser.add_argument('-train_kb_path', type=str, help='model type')
	parser.add_argument('-valid_kb_path', type=str, help='model type')	
	parser.add_argument('-train_celeb_path', type=str, help='celeb path')
	parser.add_argument('-valid_celeb_path', type=str, help='celeb path')
	parser.add_argument('-celeb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-kb_vocab_path', type=str, help='celeb path')
	parser.add_argument('-num_states', type=int, help='num states for multitasking')
	args = parser.parse_args()
	main(args)
