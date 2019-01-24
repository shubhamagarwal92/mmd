#!/usr/bin/env python
# # -*- coding: utf-8 -*-
""" Encoder for Sequence to Sequence models """
__author__ = "shubhamagarwal92"

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch_utils as torch_utils

class EncoderRNN(nn.Module):
	r""" Basic encoder cell in encoder decoder framework
	Args:
		enc_vocab_size (int): size of input vocab 
			(Used to define embedding layer)
		enc_emb_size (int): size of embedding layer
			(Usually omitted and assumed equal to enc_hidden_size)
		enc_hidden_size (int): hidden size of each layer in Encoder RNN 
			(Generally same as Decoder RNN)
		rnn_type (str): 
			type of rnn [LSTM, GRU] (default: GRU)
			nn.LSTM and nn.GRU are used instead of nn.LSTMCell/GRUCell
			input needs to be 'packed' before usage
			to handle variable length input sequences
		num_layers (int) : number of layers (default:1)
		batch_first (bool): (default: True)
			input and output tensors are provided as (batch, seq, feature)
		dropout (float) : dropout value (default:0)
		bidirectional (bool) : use a bidirectional RNN (default: True)
	Input:
		input_seq: (batch,seq_length) **Sorted** by seq_length for pack_padded_sequence
		seq_lengths: sequence lengths; needed for pack_padded_sequence
		hidden: (Default:None) initial Hidden state of RNN 
			(layers*directions,batch_size,enc_hidden_size) 
			https://github.com/pytorch/pytorch/issues/434
	"""
	def __init__(self, enc_vocab_size, enc_emb_size, enc_hidden_size, 
				rnn_type='GRU', num_layers=1, batch_first=True,
				dropout=0, bidirectional=True):
		super(EncoderRNN, self).__init__()
		self.enc_hidden_size = enc_hidden_size
		self.batch_first = batch_first
		self.num_layers = num_layers
		# self.num_directions = 2 if self.bidirectional else 1 # In case if we half encoder size
		self.embedding = nn.Embedding(enc_vocab_size,enc_emb_size)
		# Warapper to handle both LSTM and GRU
		self.rnn_cell = torch_utils.rnn_cell_wrapper(rnn_type)
		self.encoder = self.rnn_cell(enc_emb_size, enc_hidden_size, num_layers = num_layers, 
									batch_first=batch_first, dropout=dropout, 
									bidirectional=bidirectional)

	def forward(self, input_seq, seq_length, hidden = None):
		sorted_lens, len_ix = seq_length.sort(0, descending=True)
		# https://discuss.pytorch.org/t/solved-multiple-packedsequence-input-ordering/2106
		# https://discuss.pytorch.org/t/rnns-sorting-operations-autograd-safe/1461
		# https://github.com/ctr4si/A-Hierarchical-Latent-Structure-for-Variational\
		# -Conversation-Modeling/blob/master/model/layers/encoder.py#L182
		# Used for later reorder
		inv_ix = len_ix.clone()
		inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)
		sorted_inputs = input_seq[len_ix].contiguous()
        # # Sort in decreasing order of length for pack_padded_sequence()
        # input_length_sorted, indices = input_length.sort(descending=True)
        # input_length_sorted = input_length_sorted.data.tolist()
        # # [num_sentences, max_source_length]
        # inputs_sorted = inputs.index_select(0, indices)

		# Embedding layer
		embedded = self.embedding(sorted_inputs) # input_seq = (batch,seq_length)
		# pack_padded_sequence for variable length
		# Without sorting
		# embedded = self.embedding(input_seq) # input_seq = (batch,seq_length)
		# packed_embbed = pack(embedded, seq_length, batch_first=self.batch_first)
		packed_embbed = pack(embedded, list(sorted_lens.data), batch_first=self.batch_first)
		# Encoder RNN output for packed sequence (for batch_first flag)
		# No need to provide hidden initial state
		# https://github.com/pytorch/pytorch/issues/434
		# output (batch, seq_len, enc_hidden_size * num_directions)	=> default (batch,len,dim)
		# hidden (num_layers * num_directions, batch, enc_hidden_size) => default(1*2,batch,dim)
		# self.encoder.flatten_parameters()
		output, hidden = self.encoder(packed_embbed)
		# If we want to provide initial hidden state use 
		# output, hidden = self.encoder(packed_embbed, hidden)
		output, output_length = unpack(output, batch_first=self.batch_first)
        # Reorder outputs and hidden
        # _, inverse_indices = indices.sort()
        # outputs = outputs.index_select(0, inverse_indices)
		output = output[inv_ix].contiguous()
		hidden = hidden[:, inv_ix.data, ].contiguous()
		return output, hidden
	