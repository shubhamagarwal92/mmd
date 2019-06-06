import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from encoderRNN import EncoderRNN
from image_encoder import ImageEncoder
from bridge import BridgeLayer
from contextRNN import ContextRNN
from decoder import DecoderRNN
import torch_utils as torch_utils
from torch_utils import to_var
from kb_encoder import KbEncoder

class MultimodalHRED(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=True, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=1, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity='tanh', enc_type='GRU', dec_type='GRU', context_type='GRU', 
				use_attention=True, decode_function='softmax', sos_id=2, eos_id=3, 
				tie_embedding=True, activation_bridge='Tanh', num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None):
		super(MultimodalHRED, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.src_emb_dim = src_emb_dim
		self.tgt_emb_dim = tgt_emb_dim
		self.batch_size = batch_size
		self.bidirectional_enc = bidirectional_enc
		self.bidirectional_context = bidirectional_context
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.num_context_layers = num_context_layers
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.dropout_context = dropout_context #dropout prob for context
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.enc_type = enc_type
		self.dec_type = dec_type
		self.context_type = context_type
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.decode_function = decode_function # @TODO: softmax or log softmax 
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size // self.num_directions
		self.num_directions = 2 if bidirectional_context else 1
		self.context_hidden_size = context_hidden_size // self.num_directions
		self.dec_hidden_size = dec_hidden_size
		self.use_attention = use_attention
		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		## TODO - copy this to all
		self.use_kb = use_kb 
		self.kb_size = kb_size 
		self.celeb_vec_size = celeb_vec_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size
		if self.use_kb:
			self.kb_encoder = KbEncoder(self.kb_size, self.kb_emb_size, self.kb_hidden_size,
			                	rnn_type='GRU', num_layers=1, batch_first=True,
			                	dropout=0, bidirectional=False)
			# Same for kb and celebs for now.
			self.celeb_encoder = KbEncoder(self.celeb_vec_size, self.kb_emb_size, self.kb_hidden_size,
			                	rnn_type='GRU', num_layers=1, batch_first=True,
			                	dropout=0, bidirectional=False)

		# Initialize encoder
		self.encoder = EncoderRNN(self.src_vocab_size, self.src_emb_dim, self.enc_hidden_size, 
						self.enc_type, self.num_enc_layers, batch_first=True, dropout=self.dropout_enc, 
						bidirectional=self.bidirectional_enc)
		self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)
		# Initialize bridge layer 
		self.activation_bridge = activation_bridge
		# self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size, self.activation_bridge)
		self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size)
		# Initialize context encoder
		self.context_input_size = self.image_out_size + enc_hidden_size # image+text
		self.context_encoder = ContextRNN(self.context_input_size, self.context_hidden_size, 
								self.context_type, self.num_context_layers, batch_first=True,
								dropout=self.dropout_context, bidirectional=self.bidirectional_context)
		# Initialize RNN decoder
		self.decoder = DecoderRNN(self.tgt_vocab_size, self.tgt_emb_dim, self.dec_hidden_size, 
						self.dec_type, self.num_dec_layers, self.max_decode_len,  
						self.dropout_dec, batch_first=True, use_attention=self.use_attention, 
						attn_size = self.attention_size, sos_id=self.sos_id, eos_id=self.eos_id,
						use_input_feed=True,
						use_kb=self.use_kb, kb_size=self.kb_hidden_size, celeb_vec_size=self.kb_hidden_size)						
		if tie_embedding:
			self.decoder.embedding = self.encoder.embedding
		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array
		assert (text_enc_input.size(0)==context_size), "Context size not equal to first dimension"
		# Define variables to store outputs
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		context_enc_input_in_place = Variable(torch.zeros(batch_size, context_size, \
							self.dec_hidden_size*2), requires_grad=True)
		# https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/6
		# https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/6
		context_enc_input = context_enc_input_in_place.clone()
		context_enc_input = torch_utils.gpu_wrapper(context_enc_input, use_cuda=use_cuda) # Port to cuda
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input
			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
			# Bridge layer to pass encoder outputs to context RNN # (layers*directions, batch, features)
			# [-1] => (B,D); select the last, unsqueeze to (B,1,D)
			text_outputs = self.bridge(encoder_hidden, 
								bidirectional_encoder=self.bidirectional_enc)[-1] # (B,dim)

			image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images =1, features=4096*5=in_size)
			image_outputs = self.image_encoder(image_input).squeeze(1)
			# image_outputs = image_outputs.contiguous() # Error in py2
			combined_enc_input = self.combine_enc_outputs(text_outputs, image_outputs, dim=1)
			context_enc_input[:,turn,:] = combined_enc_input # (batch, 1, features)
		# Context RNN	
		context_enc_outputs, context_enc_hidden = self.context_encoder(context_enc_input)
		context_projected_hidden = self.bridge(context_enc_hidden, 
								bidirectional_encoder=self.bidirectional_context)#.unsqueeze(0) 
								# (B,D) => (Layer,B,D)
		# TODO: copy here and in decode as well.
		kb_outputs = None
		celeb_outputs = None
		if self.use_kb:
			_, kb_hidden = self.kb_encoder(kb_vec, kb_len)
			kb_outputs = self.bridge(kb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)
			_, celeb_hidden = self.celeb_encoder(celeb_vec, celeb_len)
			celeb_outputs = self.bridge(celeb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)
		if not decode:
			decoder_outputs = self.decoder(dec_text_input,
								init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			return decoder_outputs
		else:
			prediction = self.decoder(init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			return prediction

	def combine_enc_outputs(self, text_outputs, image_outputs, dim=2):
		"""Combine tensors across specified dimension. """
		encoded_both = torch.cat([image_outputs, text_outputs],dim)
		return encoded_both

	def softmax_prob(self, logits):
		"""Return probability distribution over words."""
		soft_probs = torch_utils.softmax_3d(logits)
		return soft_probs

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.requires_grad:
				param.data.uniform_(-initrange, initrange)

class HRED(nn.Module):
	r""" HRED model
	Args:
	Inputs: 
	Outputs: 
	"""
	def __init__(self, src_vocab_size, tgt_vocab_size, src_emb_dim, tgt_emb_dim, 
				enc_hidden_size, dec_hidden_size, context_hidden_size, batch_size, 
				image_in_size, bidirectional_enc=True, bidirectional_context=False, 
				num_enc_layers=1, num_dec_layers=1, num_context_layers=1, 
				dropout_enc=0.4, dropout_dec=0.4, dropout_context=0.4, max_decode_len=40, 
				non_linearity='tanh', enc_type='GRU', dec_type='GRU', context_type='GRU', 
				use_attention=True, decode_function='softmax', sos_id=2, eos_id=3, 
				tie_embedding=True, activation_bridge='Tanh', num_states=None,
				use_kb=False, kb_size=None, celeb_vec_size=None):
		super(HRED, self).__init__()
		self.src_vocab_size = src_vocab_size
		self.tgt_vocab_size = tgt_vocab_size
		self.src_emb_dim = src_emb_dim
		self.tgt_emb_dim = tgt_emb_dim
		self.batch_size = batch_size
		self.bidirectional_enc = bidirectional_enc
		self.bidirectional_context = bidirectional_context
		self.num_enc_layers = num_enc_layers
		self.num_dec_layers = num_dec_layers
		self.num_context_layers = num_context_layers
		self.dropout_enc = dropout_enc #dropout prob for encoder
		self.dropout_dec = dropout_dec #dropout prob for decoder
		self.dropout_context = dropout_context #dropout prob for context
		self.non_linearity = non_linearity # default nn.tanh(); nn.relu()
		self.enc_type = enc_type
		self.dec_type = dec_type
		self.context_type = context_type
		self.sos_id = sos_id # start token		
		self.eos_id = eos_id # end token
		self.decode_function = decode_function # @TODO: softmax or log softmax 
		self.max_decode_len = max_decode_len # max timesteps for decoder
		self.attention_size = dec_hidden_size # Same as enc/dec hidden size!!
		# self.context_hidden_size = context_hidden_size
		# self.enc_hidden_size = enc_hidden_size
		# All implementations have encoder hidden size halved
		self.num_directions = 2 if bidirectional_enc else 1
		self.enc_hidden_size = enc_hidden_size // self.num_directions
		self.num_directions = 2 if bidirectional_context else 1
		self.context_hidden_size = context_hidden_size // self.num_directions
		self.dec_hidden_size = dec_hidden_size
		self.use_attention = use_attention
		self.image_in_size = image_in_size
		self.image_out_size = self.dec_hidden_size # Project on same size as enc hidden

		self.use_kb = use_kb 
		self.kb_size = kb_size 
		self.celeb_vec_size = celeb_vec_size
		# Equating to emb_size = tgt_emb_dim for now 
		# Default to hidden_size = dec_hidden_size for now. 
		self.kb_emb_size = self.tgt_emb_dim
		self.kb_hidden_size = self.dec_hidden_size
		if self.use_kb:
			self.kb_encoder = KbEncoder(self.kb_size, self.kb_emb_size, self.kb_hidden_size,
				                rnn_type='GRU', num_layers=1, batch_first=True,
			    	            dropout=0, bidirectional=False)
			# Same for kb and celebs for now.
			self.celeb_encoder = KbEncoder(self.celeb_vec_size, self.kb_emb_size, self.kb_hidden_size,
				                rnn_type='GRU', num_layers=1, batch_first=True,
			    	            dropout=0, bidirectional=False)


		# Initialize encoder
		self.encoder = EncoderRNN(self.src_vocab_size, self.src_emb_dim, self.enc_hidden_size, 
						self.enc_type, self.num_enc_layers, batch_first=True, dropout=self.dropout_enc, 
						bidirectional=self.bidirectional_enc)
		# self.image_encoder = ImageEncoder(self.image_in_size, self.image_out_size)
		# Initialize bridge layer 
		self.activation_bridge = activation_bridge
		# self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size, self.activation_bridge)
		self.bridge = BridgeLayer(self.enc_hidden_size, self.dec_hidden_size)
		# Initialize context encoder
		self.context_input_size = enc_hidden_size #self.image_out_size + enc_hidden_size # image+text
		self.context_encoder = ContextRNN(self.context_input_size, self.context_hidden_size, 
								self.context_type, self.num_context_layers, batch_first=True,
								dropout=self.dropout_context, bidirectional=self.bidirectional_context)
		# Initialize RNN decoder
		self.decoder = DecoderRNN(self.tgt_vocab_size, self.tgt_emb_dim, self.dec_hidden_size, 
						self.dec_type, self.num_dec_layers, self.max_decode_len,  
						self.dropout_dec, batch_first=True, use_attention=self.use_attention, 
						attn_size = self.attention_size, sos_id=self.sos_id, eos_id=self.eos_id,
						use_input_feed=True,
						use_kb=self.use_kb, kb_size=self.kb_hidden_size, celeb_vec_size=self.kb_hidden_size)						

		if tie_embedding:
			self.decoder.embedding = self.encoder.embedding
		# Initialize parameters
		self.init_params()

	def forward(self, text_enc_input, image_enc_input, text_enc_in_len=None, dec_text_input=None,
				dec_out_seq=None, context_size=2, teacher_forcing_ratio=1, decode=False, 
				use_cuda=False, beam_size=1, kb_vec=None, celeb_vec=None, kb_len=None, celeb_len=None):
		# text_enc_input == (turn, batch, seq_len) ==> will project it to features through RNN
		# text_enc_in_len == (turn, batch) # np.array
		assert (text_enc_input.size(0)==context_size), "Context size not equal to first dimension"
		# Define variables to store outputs
		batch_size = text_enc_input.size(1)
		# https://github.com/pytorch/pytorch/issues/5552
		context_enc_input_in_place = Variable(torch.zeros(batch_size, context_size, \
							self.context_input_size), requires_grad=True)
		# https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/6
		# https://discuss.pytorch.org/t/how-to-copy-a-variable-in-a-network-graph/1603/6
		context_enc_input = context_enc_input_in_place.clone()
		context_enc_input = torch_utils.gpu_wrapper(context_enc_input, use_cuda=use_cuda) # Port to cuda
		for turn in range(0,context_size):
			# pytorch multiple packedsequence input ordering
			text_input = text_enc_input[turn,:] #3D to 2D (batch, seq_len) == regular input

			# Pass through encoder: 
			# text_enc_in_len[turn,:] # 2D to 1D
			encoder_outputs, encoder_hidden = self.encoder(text_input, text_enc_in_len[turn])
			# Bridge layer to pass encoder outputs to context RNN # (layers*directions, batch, features)
			text_outputs = self.bridge(encoder_hidden, bidirectional_encoder=self.bidirectional_enc)[-1] # (B,dim)
			# image_input = image_enc_input[turn,:] #4D to 3D (batch, seq_len = num_images, features)
			# image_outputs = self.image_encoder(image_input).squeeze(1)
			# image_outputs = image_outputs.contiguous() # Error in py2
			# combined_enc_input = self.combine_enc_outputs(text_outputs, image_outputs, dim=1)
			context_enc_input[:,turn,:] = text_outputs # (batch, 1, features)
		# Context RNN	
		context_enc_outputs, context_enc_hidden = self.context_encoder(context_enc_input)
		context_projected_hidden = self.bridge(context_enc_hidden, 
								bidirectional_encoder=self.bidirectional_context)#.unsqueeze(0) 
								# (B,D) => (Layer,B,D)
		kb_outputs = None
		celeb_outputs = None
		if self.use_kb:
			_, kb_hidden = self.kb_encoder(kb_vec, kb_len)
			kb_outputs = self.bridge(kb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)
			_, celeb_hidden = self.celeb_encoder(celeb_vec, celeb_len)
			celeb_outputs = self.bridge(celeb_hidden, 
					bidirectional_encoder=False)[-1] # (B,dim)

		if not decode:
			decoder_outputs = self.decoder(dec_text_input,
										   init_h=context_projected_hidden,
										   encoder_outputs = encoder_outputs,
										   input_valid_length = text_enc_in_len[turn],
										   context_enc_outputs = context_enc_outputs,
										    kb_vec = kb_outputs,
										    celeb_vec = celeb_outputs, 
											decode=decode)
			return decoder_outputs
		else:
			prediction = self.decoder(init_h=context_projected_hidden,
								encoder_outputs = encoder_outputs,
								input_valid_length = text_enc_in_len[turn],
								context_enc_outputs = context_enc_outputs,
							    kb_vec = kb_outputs,
							    celeb_vec = celeb_outputs, 
								decode=decode)
			# prediction, final_score, length = self.decoder.beam_decode(beam_size=beam_size, 
			# 									init_h=context_projected_hidden,
			# 								   encoder_outputs = encoder_outputs,
			# 								   input_valid_length = text_enc_in_len[turn],
			# 								   context_enc_outputs = context_enc_outputs,
			# 								   kb_vec = kb_outputs,
			# 								   celeb_vec = celeb_outputs)
			return prediction

	def init_params(self, initrange=0.1):
		for name, param in self.named_parameters():
			if param.requires_grad:
				param.data.uniform_(-initrange, initrange)
