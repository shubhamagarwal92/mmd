# Contains the wrappers for torch functions, useful for model 
import torch
import torch.nn as nn
from torch.autograd import Variable

def to_var(x, on_cpu=False):
	"""Tensor => Variable"""
	x = Variable(x)
	if torch.cuda.is_available() and not on_cpu:
		x = gpu_wrapper(x)
	return x

def object_type(obj):
	""" Return whether the object is tensor or variable """
	if isinstance(obj, torch.Tensor):
		return "tensor"
	if isinstance(obj,Variable):
		return "Variable"

def gpu_wrapper(input_var, use_cuda=True):
	""" Port variable/tensor to gpu """
	if use_cuda:
		input_var = input_var.cuda()
	return input_var

def rnn_cell_wrapper(rnn_type='GRU'):
	""" Wrapper to handle different gating mechanisms """
	if rnn_type == 'GRU':
		rnn_cell = nn.GRU
	elif rnn_type == 'LSTM':
		rnn_cell = nn.LSTM
	else:
		raise Exception('Unknown rnn cell: {}'.format(rnn_type))
	return rnn_cell

def sequence_mask(lengths, max_len=None):
	"""
	Creates a boolean mask from sequence lengths.
	"""
	batch_size = lengths.numel()
	max_len = max_len or lengths.max()
	return (torch.arange(0, max_len)
			.type_as(lengths)
			.repeat(batch_size, 1)
			.lt(lengths.unsqueeze(1)))

def linear_3d(input_variable, linear_layer):
	"""
	Use as:
	linear_layer = nn.Linear(initial_dim, projected_dim, bias=False)
	input_variable (batch, len, features)
	Transform from 3D to 2D, apply linear and return 3D
	"""
	# If applying softmax, we can directly use size for view 
	# while projecting back bcoz softmax doesn't change the dimension
	size = input_variable.size()
	input_2d = input_variable.view(-1, size[2])
	result_2d = linear_layer(input_2d)
	result_3d = result_2d.view(size[0], size[1], -1)  
	return result_3d

def softmax_3d(input_variable, softmax_layer):
	"""
	softmax_layer = nn.Softmax(dim=-1) # or nn.LogSoftmax(dim=-1)
	input_variable (batch, len, features)
	# Need to reshape just to apply softmax 
	# https://github.com/pytorch/pytorch/issues/1020
	# https://github.com/pytorch/pytorch/issues/1763
	Transform from 3D to 2D, apply softmax and return 3D
	"""
	# If applying softmax, we can directly use size for view 
	# while projecting back bcoz softmax doesn't change the dimension
	size = input_variable.size()
	input_2d = input_variable.view(-1, size[2])
	result_2d = softmax_layer(input_2d)
	# Reshape after normalize
	# result_3d = result_2d.view(size[0], size[1], -1)  
	result_3d = result_2d.view(size) 
	return result_3d