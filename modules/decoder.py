# Adapted from https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-\
# Variational-Conversation-Modeling/blob/master/model/layers/decoder.py

import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch_utils import to_var
import math
from attention import AttentionLayer

class BaseRNNDecoder(nn.Module):
	def __init__(self):
		"""Base Decoder Class"""
		super(BaseRNNDecoder, self).__init__()

	def _input_size(self):
		"""
		Using input feed by concatenating embedded input with 
		attention vectors from previous time step.
		"""
		_in_size = self.embedding_size
		# Use input feeding
		if self.use_input_feed:
			_in_size = _in_size + self.hidden_size
		# Use kb
		if self.use_kb:
			_in_size = _in_size + self.kb_size + self.celeb_vec_size
		# Use state vector
		if self.is_mutlitask:
			_in_size = _in_size + self.state_size
		return _in_size

	@property
	def use_lstm(self):
		return isinstance(self.rnncell, nn.LSTM)

	def init_token(self, batch_size, sos_id=2):
		"""Get Variable of <SOS> Index (batch_size)"""
		x = to_var(torch.LongTensor([sos_id] * batch_size))
		return x

	def init_h(self, batch_size=None, zero=True, hidden=None):
		"""Return RNN initial state"""
		if hidden is not None:
			return hidden

		if self.use_lstm:
			# (h, c)
			return (to_var(torch.zeros(self.num_layers,
									   batch_size,
									   self.hidden_size)),
					to_var(torch.zeros(self.num_layers,
									   batch_size,
									   self.hidden_size)))
		else:
			# h
			return to_var(torch.zeros(self.num_layers,
									  batch_size,
									  self.hidden_size))
	def init_feed(self, hidden, size):
		# New creates a new variable of the same type as given tensor
		# but of specified size.
		input_feed = Variable(hidden[0].data.new(*size).zero_()).unsqueeze(1)
		return input_feed


	def batch_size(self, inputs=None, h=None):
		"""
		inputs: [batch_size, seq_len]
		h: [num_layers, batch_size, hidden_size] (RNN/GRU)
		h_c: [2, num_layers, batch_size, hidden_size] (LSTMCell)
		"""
		if inputs is not None:
			batch_size = inputs.size(0)
			return batch_size

		else:
			if self.use_lstm:
				batch_size = h[0].size(1)
			else:
				batch_size = h.size(1)
			return batch_size

	def decode(self, out):
		"""
		Args:
			out: unnormalized word distribution [batch_size, vocab_size]
		Return:
			x: word_index [batch_size]
		"""

		# Sample next word from multinomial word distribution
		if self.sample:
			# x: [batch_size] - word index (next input)
			x = torch.multinomial(self.softmax(out / self.temperature), 2).view(-1)
		# Greedy sampling
		else:
			# x: [batch_size] - word index (next input)
			_, x = out.max(dim=2)
		return x

	def forward(self):
		"""Base forward function to inherit"""
		raise NotImplementedError

	def forward_step(self):
		"""Run RNN single step"""
		raise NotImplementedError

	def embed(self, x):
		embed = self.embedding(x)
		return embed


class DecoderRNN(BaseRNNDecoder):
	def __init__(self, vocab_size, embedding_size,
				 hidden_size, rnncell='GRU', num_layers=1,
				 max_unroll=40, dropout=0.0, word_drop=0.0, 
				 batch_first=True, sample=False, temperature=1.0, 
				 use_attention=True, attn_size = 128, 
				 sos_id=2, eos_id=3, use_input_feed=True, 
				 use_kb=False, is_mutlitask=False,
				 kb_size=None, celeb_vec_size=None, state_size=None):
		super(DecoderRNN, self).__init__()

		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.sos_id = sos_id
		self.eos_id = eos_id
		self.num_layers = num_layers
		self.dropout = dropout
		self.temperature = temperature
		self.word_drop = word_drop
		self.max_unroll = max_unroll
		self.sample = sample
		self.is_mutlitask = is_mutlitask
		self.use_kb = use_kb
		self.state_size = state_size
		self.kb_size = kb_size
		# self.beam_size = beam_size
		self.attn_size = attn_size
		self.celeb_vec_size = celeb_vec_size
		self.use_input_feed = use_input_feed
		self.embedding = nn.Embedding(vocab_size, self.embedding_size)
		self.rnncell = nn.GRU(self._input_size(), self.hidden_size, num_layers = num_layers, 
									batch_first=batch_first, dropout=dropout)
		self.use_attention = use_attention
		self.attention = AttentionLayer(self.attn_size)
		self.out = nn.Linear(hidden_size, vocab_size)
		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

	def forward_step(self, x, h, 
					 encoder_outputs=None,
					 input_valid_length=None, input_feed=None, 
					 context_enc_outputs = None,
					 image_outputs= None,
					 kb_vec= None,
					 celeb_vec = None, 
					 state_vec=None):
		"""
		Single RNN Step
		1. Input Embedding (vocab_size => hidden_size)
		2. RNN Step (hidden_size => hidden_size)
		3. Output Projection (hidden_size => vocab size)

		Args:
			x: [batch_size]
			h: [num_layers, batch_size, hidden_size] (h and c from all layers)

		Return:
			out: [batch_size,vocab_size] (Unnormalized word distribution)
			h: [num_layers, batch_size, hidden_size] (h and c from all layers)
		"""
		decoder_input = self.embed(x.unsqueeze(1))
		if self.use_input_feed:
			# input_feed: [batch_size, hidden_size] => [batch_size, 1, hidden_size]
			# input_feed = input_feed.unsqueeze(1)
			decoder_input = torch.cat([decoder_input, input_feed], 2)
		# else:
		# 	decoder_input = x
		if self.is_mutlitask:
			# state_vec: [batch_size, state_size] => [batch_size, 1, state_size]
			state_vec = self.sigmoid(state_vec.unsqueeze(1))
			decoder_input = torch.cat([decoder_input, state_vec], 2)
		if self.use_kb:
			# state_vec: [batch_size, state_size] => [batch_size, 1, state_size]
			kb_vec = kb_vec.unsqueeze(1)
			celeb_vec = celeb_vec.unsqueeze(1)
			combined_kb_vec = torch.cat([kb_vec, celeb_vec], 2)
			decoder_input = torch.cat([decoder_input, combined_kb_vec], 2)

		# last_h: [batch_size, hidden_size] (h from Top RNN layer)
		# h: [num_layers, batch_size, hidden_size] (h and c from all layers)

		last_h, h = self.rnncell(decoder_input, h)
		# if self.use_lstm:
		#     # last_h_c: [2, batch_size, hidden_size] (h from Top RNN layer)
		#     # h_c: [2, num_layers, batch_size, hidden_size] (h and c from all layers)
		#     last_h = last_h[0]
		if self.use_attention:
			attn_vec_cxt, attn_wts_cxt = self.attention(last_h, context_enc_outputs)
			out = self.out(attn_vec_cxt)
			return out, h, attn_vec_cxt
		else:
			out = self.out(last_h)
			return out, h, last_h

	def forward(self, inputs=None, init_h=None, encoder_outputs=None, 
				input_valid_length=None, context_enc_outputs = None, 
				image_outputs = None, kb_vec = None, celeb_vec = None, 
				state_vec= None, decode=False):
		"""
		Train (decode=False)
			Args:
				inputs (Variable, LongTensor): [batch_size, seq_len]
				init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
			Return:
				out   : [batch_size, seq_len, vocab_size]
		Test (decode=True)
			Args:
				inputs: None
				init_h: (Variable, FloatTensor): [num_layers, batch_size, hidden_size]
			Return:
				out   : [batch_size, seq_len]
		"""
		batch_size = self.batch_size(inputs, init_h)

		# x: [batch_size]
		x = self.init_token(batch_size, self.sos_id)

		# h: [num_layers, batch_size, hidden_size]
		h = self.init_h(batch_size, hidden=init_h)
		size = (h.size(1),h.size(2))
		input_feed = self.init_feed(h,size)
		if not decode:
			out_list = []
			seq_len = inputs.size(1)
			for i in range(seq_len):
				x = inputs[:, i]
				out, h, input_feed = self.forward_step(x, h, encoder_outputs, 
										input_valid_length, input_feed,
										context_enc_outputs, image_outputs,
										kb_vec, celeb_vec, state_vec)
				out_list.append(out)
			return torch.cat(out_list, dim=1)
		else:
			x_list = []
			for i in range(self.max_unroll):
				out, h, input_feed = self.forward_step(x, h, encoder_outputs,
										input_valid_length, input_feed,
										context_enc_outputs, image_outputs,
										kb_vec, celeb_vec, state_vec)
				x = self.decode(out).squeeze(1)
				x_list.append(x.unsqueeze(1))
			pred = torch.cat(x_list, dim=1).unsqueeze(1)
			return pred
