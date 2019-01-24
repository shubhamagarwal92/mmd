import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_utils as torch_utils

class ImageEncoder(nn.Module):
	r"""

	Args:
		
	Input:
		
	Output:
		
	"""
	def __init__(self, image_in_size, image_out_size, bias=False, activation='Tanh'):
		super(ImageEncoder, self).__init__()
		self.input_size = image_in_size
		self.output_size = image_out_size
		self.image_proj_layer = nn.Linear(self.input_size,self.output_size, bias=bias)
		# self.activation = getattr(nn, activation)()

	def forward(self, image_input):
		image_outputs = torch_utils.linear_3d(image_input, self.image_proj_layer)
		# enc_final_hidden = self.activation(enc_final_hidden)
		image_outputs = F.relu(image_outputs)

		return image_outputs