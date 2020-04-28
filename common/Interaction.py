import torch
import torch.nn as nn
import torch.nn.functional as F

class Interaction(nn.Module):
    def __init__(self, hidden_size):
        super(Interaction, self).__init__()
        self.hidden_size=hidden_size

        # self.m1 = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)
        # self.m2 = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)
        # self.m3 = nn.Bilinear(hidden_size, hidden_size, 1, bias=False)
        self.dual_att_linear=nn.Linear(3*hidden_size, 1, bias=False)

    def forward(self, encode_input1, encode_input2, input1_mask, input2_mask):
        """
        :return: [batch_size, 1 or num_sequences, input_1_sequence_len, 5*hidden_size], [batch_size, num_sequences, input_2_sequence_len, 5*hidden_size]
        """
        # E_q_=encode_input1#[batch_size, num_sequences=1, seq_len_q, hidden_size]
        # E_p=encode_input2#[batch_size, num_sequences, seq_len_p, hidden_size]

        batch_size, num_q, seq_len_q, hidden_size = encode_input1.size()
        batch_size, num_p, seq_len_p, hidden_size = encode_input2.size()
        if num_q==num_p:
            E_q = encode_input1.reshape(-1, seq_len_q, hidden_size)  # [batch_size*num_sequences, seq_len_q, hidden_size]
        else:
            assert num_q==1
            E_q = encode_input1.expand(-1, num_p, -1, -1).reshape(-1, seq_len_q, hidden_size)  # [batch_size*num_sequences, seq_len_q, hidden_size]
            input1_mask=input1_mask.expand(-1, num_p, -1)
        E_p=encode_input2.reshape(-1, seq_len_p, hidden_size)#[batch_size*num_sequences, seq_len_p, hidden_size]

        E_q_temp=E_q.unsqueeze(1).expand(-1, seq_len_p, -1, -1).contiguous()
        E_p_temp=E_p.unsqueeze(2).expand(-1, -1, seq_len_q, -1).contiguous()

        E=torch.cat([E_q_temp, E_p_temp, E_q_temp*E_p_temp], dim=-1)#[batch_size*num_sequences, seq_len_p, seq_len_q, 3*hidden_size]
        U = self.dual_att_linear(E).squeeze(-1)  # [batch_size*num_sequences, seq_len_p, seq_len_q]
        # E_q_p_temp=E_q_temp*E_p_temp
        # U=(self.m1(E_q_temp, E_p_temp)+self.m2(E_q_temp, E_q_p_temp)+self.m3(E_p_temp, E_q_p_temp)).squeeze(-1)

        # if input1_mask is not None and input2_mask is not None:
        mask=torch.bmm(input2_mask.reshape(-1, seq_len_p).unsqueeze(2).float(), input1_mask.reshape(-1, seq_len_q).unsqueeze(1).float())#[batch_size*num_sequences, seq_len_p, seq_len_q]
        U=U.masked_fill(~mask.bool(), -float('inf'))
        A_p = F.softmax(U, dim=2)#[batch_size*num_sequences, seq_len_p, seq_len_q]
        B_p = F.softmax(U, dim=1)#[batch_size*num_sequences, seq_len_p, seq_len_q]
        # if input1_mask is not None and input2_mask is not None:
        A_p = A_p.masked_fill(~mask.bool(), 0)
        B_p = B_p.masked_fill(~mask.bool(), 0)

        A__p=torch.bmm(A_p, E_q)#[batch_size*num_sequences, seq_len_p, hidden_size]
        B__p=torch.bmm(B_p.transpose(1,2), E_p)#[batch_size*num_sequences, seq_len_q, hidden_size]

        A___p=torch.bmm(A_p, B__p)#[batch_size*num_sequences, seq_len_p, hidden_size]
        B___p=torch.bmm(B_p.transpose(1,2), A__p)#[batch_size*num_sequences, seq_len_q, hidden_size]

        E_p = E_p.reshape(batch_size, num_p, seq_len_p, hidden_size)
        E_q = E_q.reshape(batch_size, num_p, seq_len_q, hidden_size)
        # E_q=E_q.reshape(batch_size, num_p, seq_len_q, hidden_size)
        # A_p=A_p.reshape(batch_size, num_p, seq_len_p, seq_len_q)
        # B_p=B_p.reshape(batch_size, num_p, seq_len_p, seq_len_q)
        A__p=A__p.reshape(batch_size, num_p, seq_len_p, hidden_size)
        B__p=B__p.reshape(batch_size, num_p, seq_len_q, hidden_size)
        A___p=A___p.reshape(batch_size, num_p, seq_len_p, hidden_size)
        B___p=B___p.reshape(batch_size, num_p, seq_len_q, hidden_size)

        G_q_p=torch.cat([E_p, A__p, A___p, E_p*A__p, E_p*A___p], dim=-1)
        G_p_q=torch.cat([E_q, B__p, B___p, E_q*B__p, E_q*B___p], dim=-1)

        # if input1_mask is not None:
        G_p_q = G_p_q.masked_fill(~input1_mask.bool().unsqueeze(-1), 0)
        # if input2_mask is not None:
        G_q_p = G_q_p.masked_fill(~input2_mask.bool().unsqueeze(-1), 0)

        if num_q != num_p:
            G_p_q=G_p_q.max(dim=1, keepdim=True)[0]

        return G_p_q, G_q_p