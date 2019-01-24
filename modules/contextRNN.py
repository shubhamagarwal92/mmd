import torch
import torch.nn as nn
import torch_utils as torch_utils

class ContextRNN(nn.Module):
	r"""
	Context RNN for HRED model
	Args:
	   rnn_type (str): type of RNN [LSTM, GRU]
	   bidirectional (bool) : use a bidirectional RNN
	   num_layers (int) : number of stacked layers
	   context_hidden_size (int) : hidden size of each layer
	   dropout (float) : dropout value for nn.Dropout
	Input: 
		context_input: Outputs of final encoder RNNs 
						(batch,enc_size,enc_hidden_size)
	Output:
		context_out: context vector for decoder rnn at current time step 
	Usage:
	"""
	def __init__(self, context_input_size, context_hidden_size, rnn_type='GRU', 
				num_layers=1, batch_first=True, dropout=0, bidirectional=False):
		super(ContextRNN, self).__init__()
		# Defining parameters for reference
		self.num_layers = num_layers
		self.context_hidden_size = context_hidden_size
		self.context_input_size = context_input_size
		self.rnn_cell = torch_utils.rnn_cell_wrapper(rnn_type)
		self.contextRNN = self.rnn_cell(self.context_input_size, self.context_hidden_size,
							num_layers = num_layers, batch_first=batch_first,
							dropout=dropout, bidirectional=bidirectional)
		#@TODO if context_hidden_size != dec_hidden_size
		# self.projectionLayer = nn.Linear(context_hidden_size, dec_hidden_size)

	def forward(self, context_input, context_hidden=None):
		# Input: (batch, context_size, features)
		# Assume input to be of always same length, context size.
		# We have context_hidden as input if we want to initialize context state in future 
		context_out, hidden = self.contextRNN(context_input)
		return context_out, hidden

	#@TODO if context_hidden_size != dec_hidden_size, bridge hidden from encoder
	# def bridge_context_hidden(self, context_hidden):
	# 	self.projected_out = torch_utils.linear_3d(self.projectionLayer,context_out)
	# 	return self.projected_out
