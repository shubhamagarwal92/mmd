import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils as torch_utils

class BridgeLayer(nn.Module):
	"""
	Bridge layer is used to pass encoder final representation to decoder.
	It is not necessary that encoder and decoder have same number of hidden states.

	Activation : currently relu 
	Args:
		enc_hidden_size(int): Hidden size of encoder (E)
		dec_hidden_size(int): Hidden size of decoder (D)
	Input:
		enc_output: Final hidden representation from encoder (batch,enc_hidden)
	Output:
		dec_initial_state: (batch,1,dec_hidden)
	"""
	def __init__(self, enc_hidden_size, dec_hidden_size):
		super(BridgeLayer, self).__init__()
		self.input_size = enc_hidden_size
		self.output_size = dec_hidden_size
		self.proj_layer = nn.Linear(self.input_size,self.output_size)

	def forward(self, enc_final_hidden, enc_cell_type='GRU', bidirectional_encoder=True):
		# Check if encoder was gru or lstm. LSTM returns tuple of hidden 
		# rnn = getattr(nn, cell)()
		if enc_cell_type =='LSTM':
			enc_final_hidden = enc_final_hidden[0]
		# The encoder hidden is  (layers*directions) x batch x dim.
		# If the encoder is bidirectional, do the following transformation.
		# (layer*direction, batch, hidden_size) -> (layers, batch, directions * hidden_size)		
		if bidirectional_encoder:
			hidden_size = enc_final_hidden.size(0)
			# print h[0:] # Different from h[0,:] (which converts 3d to 2d)
			# print h[0:index:2] # slicing as in numpy (start:stop:step)
			enc_final_hidden = torch.cat([enc_final_hidden[0:hidden_size:2], \
								enc_final_hidden[1:hidden_size:2]], 2) 
		enc_final_hidden = F.relu(enc_final_hidden)
		return enc_final_hidden
