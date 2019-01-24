import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch_utils as torch_utils

class KbEncoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, 
                rnn_type='GRU', num_layers=1, batch_first=True,
                dropout=0, bidirectional=False):
        super(KbEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.num_layers = num_layers
        # In case if we half encoder size
        self.num_directions = 2 if bidirectional else 1 
        self.embedding = nn.Embedding(vocab_size,emb_size)
        # Warapper to handle both LSTM and GRU
        self.rnn_cell = torch_utils.rnn_cell_wrapper(rnn_type)
        self.encoder = self.rnn_cell(emb_size, hidden_size, 
                        num_layers = num_layers, batch_first=self.batch_first, 
                        dropout=dropout, bidirectional=bidirectional)

    def forward(self, input_seq, seq_length, hidden = None):
        sorted_lens, len_ix = seq_length.sort(0, descending=True)
        inv_ix = len_ix.clone()
        inv_ix.data[len_ix.data] = torch.arange(0, len(len_ix)).type_as(inv_ix.data)
        sorted_inputs = input_seq[len_ix].contiguous()
        # input_seq = (batch,seq_length)
        embedded = self.embedding(sorted_inputs) 
        packed_embbed = pack(embedded, list(sorted_lens.data), 
                        batch_first=self.batch_first)
        output, hidden = self.encoder(packed_embbed)
        output, output_length = unpack(output, batch_first=self.batch_first)
        output = output[inv_ix].contiguous()
        hidden = hidden[:, inv_ix.data, ].contiguous()
        return output, hidden