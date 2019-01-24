import torch
import random
import math
import time
import json
from torch.autograd import Variable
import os
import numpy as np
from annoy import AnnoyIndex

def gpu_wrapper(input_var, use_cuda=True):
	""" Port variable/tensor to gpu """
	if use_cuda:
		input_var = input_var.cuda()
	return input_var

def convert_to_tensor(data):
    data = torch.LongTensor(data)
    return data

def tensor_to_variable(x, volatile=False):
	""" Convert to torch variable """
	# if torch.cuda.is_available():
	# 	x = x.cuda()
	return Variable(x, volatile=volatile)

def read_json_config(file_path):
	"""Read JSON config"""
	json_object = json.load(open(file_path, 'r'))
	return json_object

def get_batch_mmd_data(batch_data, sos_id, eos_id, pad_id, image_rep_size, annoyIndex, annoyPkl,
						use_cuda=False, volatile=False):
	batch_data = np.array(batch_data)
	batch_size = batch_data.shape[0]
	text_enc_input = np.array(batch_data[:,0].tolist())
	text_enc_in_len = np.array(batch_data[:,1].tolist())
	dec_out_seq = np.array(batch_data[:,4].tolist())
	dec_seq_length = np.array(batch_data[:,5].tolist())
	# pad_to_target = np.reshape(np.asarray([pad_id]*batch_size), (batch_size, 1))
	# # Removing SOS and adding pad
	# dec_out_seq = np.concatenate((dec_text_input[:,1:], pad_to_target), axis=1) 
	# # Removing EOS and adding pad
	# dec_text_input[dec_text_input==eos_id]=pad_id
	# dec_out_seq doesnt contain start id. we append start id for dec_text_input
	sos_to_target = np.reshape(np.asarray([sos_id]*batch_size), (batch_size, 1))
	dec_text_input = np.concatenate((sos_to_target, dec_out_seq[:,:-1]), axis=1) 
	dec_text_input[dec_text_input==eos_id]=pad_id
	batch_image_dict = batch_data[:,2]
	image_rep = [[[get_image_representation(entry_ijk, image_rep_size, annoyIndex, annoyPkl) \
						for entry_ijk in data_dict_ij] for data_dict_ij in data_dict_i] \
						for data_dict_i in batch_image_dict]
	image_rep = np.array(image_rep)
	shape = image_rep.shape # (batch_size, context, 5, 4096)
	image_enc_input = image_rep.reshape(shape[0],shape[1],-1) # (batch_size, context, 5*4096)
	image_enc_input = np.expand_dims(image_enc_input, axis=2) # (batch_size, context, 1, 5*4096)
	text_enc_input = text_enc_input.transpose(1,0,2) # (context,batch_size,max_len)
	image_enc_input = image_enc_input.transpose(1,0,2,3) # (context,batch_size,1,4096*images)
	text_enc_in_len = text_enc_in_len.transpose(1,0) # (context,batch_size)
	text_enc_input = gpu_wrapper(tensor_to_variable(convert_to_tensor(text_enc_input),\
						volatile=volatile), use_cuda=use_cuda)
	image_enc_input = gpu_wrapper(tensor_to_variable(torch.FloatTensor(image_enc_input),\
						volatile=volatile), use_cuda=use_cuda)
	dec_out_seq = gpu_wrapper(tensor_to_variable(convert_to_tensor(dec_out_seq),\
						volatile=volatile), use_cuda=use_cuda)		
	dec_text_input = gpu_wrapper(tensor_to_variable(convert_to_tensor(dec_text_input),\
						volatile=volatile), use_cuda=use_cuda)
	text_enc_in_len = gpu_wrapper(tensor_to_variable(convert_to_tensor(text_enc_in_len),\
	 					volatile=volatile), use_cuda=use_cuda)
	dec_seq_length = gpu_wrapper(tensor_to_variable(convert_to_tensor(dec_seq_length),\
	 					volatile=volatile), use_cuda=use_cuda)
	return text_enc_input, text_enc_in_len, image_enc_input, dec_text_input, dec_out_seq, dec_seq_length

def convert_states_to_torch(data, use_cuda=False, volatile=False):
	torch_data = gpu_wrapper(tensor_to_variable(convert_to_tensor(data),\
						volatile=volatile), use_cuda=use_cuda)
	return torch_data


def get_image_representation(image_filename, image_rep_size, annoyIndex, annoyPkl):
	image_filename = image_filename.strip()	
	if image_filename=="":
		return [0.]*image_rep_size
	#FOR ANNOY BASED INDEX
	try:	
		len_images +=1
		return annoyIndex.get_item_vector(annoyPkl[image_filename])
		# Eg: 1838414, 3294309, 3469177
	except:
		return [0.]*image_rep_size
