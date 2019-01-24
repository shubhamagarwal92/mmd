# Adapted from 
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/GlobalAttention.py
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/
# seq2seq-translation-batched.ipynb
# https://github.com/google/seq2seq/blob/master/seq2seq/decoders/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils as torch_utils
from torch.autograd import Variable

class AttentionLayer(nn.Module):
	r""" Applies an attention mechanism on the output features from the decoder
		We assume that encoder outputs (aka memory) and decoder outputs (aka query)
		have been projected to dimension attention size in the decoder before the 
		AttentionLayer call
	All models compute the output as
		:math:`c = \sum_{j=1}^{SeqLength} a_j H_j` where
		:math:`a_j` is the softmax of a score function
	Then then apply a projection layer to [q, c]
	However they differ on how they compute the attention score
		- Luong Attention (dot, general):
			After the hidden vector at current timestep from RNN; 
			attention after output of RNN
			- dot: :math:`score(H_j,q) = H_j^T q`
			- general: :math:`score(H_j, q) = H_j^T W_a q`
		- Bahdanau Attention (mlp):
			Before RNN to update the hidden state; attended vector goes as input to RNN
		   - :math:`score(H_j, q) = v_a^T tanh(W_a q + U_a h_j)`
	Args:
		attn_size: Size of attention layer
		method: 
			- 'MLP': Bahdanu style MLP product attention
		query: Variable of size (B, M, attn_size)
			Batch of M query vectors
		context: Variable of size (B, N, D2)
			Batch of N context vectors
		value: Variable of size (B, N, P), default=None
			If score == 'dot', scores are computed
			as the dot product between context and
			query vectors. This Requires D1 == D2
			query    context     score
			(B,M,D1) (B,N,D2) -> (B,M,N)
			- TODO(shubhamagarwal92)
	Input: enc_output, dec_last_hidden
		enc_output: Output of encoder rnn at each time step (batch,enc_size,enc_hidden_size)
		dec_last_hidden: Output of decoder rnn at previous time step (batch,dec_size,out_vocab)
	Output:
		context_vector: context vector for decoder rnn at current time step (batch, attn_size)
		attn_vector: attention weights 
	Usage:
		>>> attention.AttentionLayer(attn_size,method='MLP')
		>>> context = Variable(torch.randn(5, 3, 256))
		>>> output = Variable(torch.randn(5, 5, 256))
		>>> output, attn = attention(output, context)
	"""
	
	def __init__(self, dim, method='dot', bidirectional_enc=True):
		super(AttentionLayer,self).__init__()
		self.dim = dim
		self.method = method
		assert (self.method in ["dot", "general", "mlp"]), (
				"Please select a valid attention type.")

		if self.method == "general":
			self.linear_general = nn.Linear(dim, dim, bias=False)
		elif self.method == "mlp":
			self.linear_enc = nn.Linear(dim, dim, bias=False)
			self.linear_dec = nn.Linear(dim, dim, bias=True)
			self.linear_mlp = nn.Linear(dim, 1, bias=False)
		# mlp wants it with bias
		out_bias = self.method == "mlp"
		self.linear_attn = nn.Linear(dim*2, dim, bias=out_bias)
		self.normalize = nn.Softmax()
		self.non_linearity = nn.Tanh()

	def forward(self, dec_output, enc_output, enc_seq_lengths=None):
		""" 
		torch_utils.linear_3d() for linear projection on 3d 
		self.score() for getting energy

		Args:
		  input (`FloatTensor`): query vectors [batch x tgt_len x dim]
		  memory_bank (`FloatTensor`): source vectors [batch x src_len x dim]
		  memory_lengths (`LongTensor`): the source context lengths [batch]
		Returns:
			Attention distribtutions for each query
			[batch, tgt_len, src_len]
		"""
		# (batch, target_length, source_length)
		energy_score = self.energy_score(enc_output, dec_output, self.method, self.dim) 
		if enc_seq_lengths is not None:
			mask = self.sequence_mask(enc_seq_lengths)
			mask = mask.unsqueeze(1)  # Make it broadcastable.
			energy_score.data.masked_fill_(1 - mask, -float('inf'))
		attention_weights = torch_utils.softmax_3d(energy_score, self.normalize)
		# Here the context vector is different from context RNN
		# (batch, target_length, source_length) * (batch, source_length, att) -> (batch, target_length, att) 
		# ==> effectively sometimes (batch, 1, att) because we are calculating at each step 
		context_vector = torch.bmm(attention_weights, enc_output) 
		# (batch, target_length, 2*att_dim)
		concat_context_hidden = torch.cat((context_vector, dec_output), dim=2) 
		# output -> (batch, target_length, att_dim)
		attention_vector = torch_utils.linear_3d(concat_context_hidden, self.linear_attn) 
		# if self.method in ["general", "dot"]: ???
			# attention_vector = self.non_linearity(attention_vector)
		attention_vector = self.non_linearity(attention_vector) # Applying non-linearity to attention vector
		# attention_vector = (batch, tgt_len, 2*attn_dim)
		# attention_weights = (batch, tgt_len, seq_len)
		return attention_vector, attention_weights

	def energy_score(self, enc_output, dec_hidden, method, dim):
		"""
		Args:
			dec_hidden: sequence of decoder [batch, tgt_len, atten_size]; tgt_len = 1 for each timestep
			enc_output: sequence of sources [batch, src_len, atten_size]
		Returns:
			raw attention scores (unnormalized) for each src index 
			[batch x tgt_len x src_len]
		"""
		src_batch, src_len, src_dim = enc_output.size() 
		tgt_batch, tgt_len, tgt_dim = dec_hidden.size()
		## src_batch == tgt_batch
		## src_dim == tgt_dim
		if self.method == "general":
			enc_output = torch_utils.linear_3d(enc_output, self.linear_general) # W_a * H_s
			enc_output = enc_output.transpose(1,2).contiguous() #pytorch/issues/764
			# batch matrix-matrix product
			# (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len, s_len)
			energy_score = torch.bmm(dec_hidden, enc_output)
			return energy_score
		elif self.method == 'dot':	
			enc_output = enc_output.transpose(1,2).contiguous()
			# batch matrix-matrix product
			# (batch, tgt_len, atten_size) x (batch, src_len, atten_size) --> (batch, tgt_len, src_len)
			energy_score = torch.bmm(dec_hidden, enc_output)			
			return energy_score
		else: # concat
			# Weighted decoder hidden
			weighted_dec_hidden = self.linear_dec(dec_hidden.view(-1, dim)) # Linear 2d
			weighted_dec_hidden = weighted_dec_hidden.view(tgt_batch, tgt_len, 1, dim) # Add src_len dimension
			weighted_dec_hidden = weighted_dec_hidden.expand(tgt_batch, tgt_len, src_len, dim)
			# Weighted encoder outputs
			weighted_enc_output = self.linear_enc(enc_output).contiguous().view(-1, dim) # Linear 2d
			weighted_enc_output = weighted_enc_output.view(src_batch, 1, src_len, dim) # Add tgt_len dimension
			weighted_enc_output = weighted_enc_output.expand(src_batch, tgt_len, src_len, dim)
			# (batch, t_len, s_len, d)
			energy_score = self.non_linearity(weighted_dec_hidden + weighted_enc_output)
			energy_score = self.linear_mlp(energy_score.view(-1, dim)).view(tgt_batch, tgt_len, src_len)
			return energy_score

	def sequence_mask(self, lengths, max_len=None):
		"""
		Creates a boolean mask from sequence lengths.
		"""
		if isinstance(lengths, Variable):
			lengths = lengths.data
		batch_size = lengths.numel()
		max_len = max_len or lengths.max()
		# .lt computes input<other element-wise.
		# torch.lt(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[1, 1], [4, 4]]))
		return (torch.arange(0, max_len)
				.type_as(lengths)
				.repeat(batch_size, 1)
				.lt(lengths.unsqueeze(1)))
