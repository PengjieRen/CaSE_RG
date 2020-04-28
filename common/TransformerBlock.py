import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
from common.Utils import *
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, input_hidden_size, output_hidden_size, activation=None):
        super(TransformerBlock, self).__init__()
        self.output_hidden_size=output_hidden_size
        self.self_attn = nn.MultiheadAttention(input_hidden_size, num_heads, dropout=0.1)
        self.norm1 = LayerNorm(input_hidden_size)
        self.norm2 = LayerNorm(input_hidden_size)
        self.linear1 = nn.Linear(input_hidden_size, output_hidden_size)
        self.linear2 = nn.Linear(output_hidden_size, output_hidden_size)

        if activation is None:
            self.activation= F.relu
        else:
            self.activation=activation

    def forward(self, input, input_mask):
        batch_size, num_seq, seq_len, hidden_size = input.size()
        reps_temp1 = input.reshape(-1, seq_len, hidden_size)
        reps_temp1_=self.norm1(reps_temp1).transpose(0, 1)
        reps_temp2 = self.self_attn(reps_temp1_, reps_temp1_,reps_temp1_, attn_mask=None, key_padding_mask=~input_mask.reshape(-1, seq_len))[0].transpose(0, 1)
        reps_temp3 = reps_temp1 + F.dropout(reps_temp2, p=0.1, training=self.training)
        reps = F.dropout(self.activation(self.linear1(self.norm2(reps_temp3))), p=0.1, training=self.training)
        reps = self.linear2(reps)

        reps=reps.reshape(batch_size, num_seq, seq_len, self.output_hidden_size)
        reps = reps.masked_fill(~input_mask.unsqueeze(-1), 0)
        return reps