import torch
import torch.nn as nn
from torch.autograd import Variable

#####################################################################################
#-------------------------------------------------------------------------------------
#Encoder
#-------------------------------------------------------------------------------------
#####################################################################################
from encoderRNN import EncoderRNN

vocab_size = 100
input_seq = Variable(torch.randperm(vocab_size).view(10, 10))  # (batch,seq_len)
print input_seq.size()
seq_length = [10,9,8,7,6,5,4,3,2,1] # This works
# seq_length = torch.LongTensor([10,9,8,7,6,5,4,3,2,1]) # This doesn't

encoder = EncoderRNN(100, 6, 2, 'GRU',1,  batch_first=True, dropout=0, 
					 bidirectional=True)
for param in encoder.parameters():
    param.data.uniform_(-1, 1)

output, hidden = encoder(input_seq, seq_length, hidden = None)

print output.size()
print hidden.size()

print output
print hidden

print type(output)
print type(hidden)




