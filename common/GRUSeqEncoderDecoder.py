import torch.nn as nn
import torch
from common.BilinearAttention import *
from common.Utils import *

class GRUSeqEncoder(nn.Module):
    def __init__(self, num_layers, src_vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super(GRUSeqEncoder, self).__init__()
        self.num_layers=num_layers

        if emb_matrix is None:
            self.embedding = nn.Embedding(src_vocab_size, embedding_size, padding_idx=0)
        else:
            self.embeddings = create_emb_layer(emb_matrix)
        self.encs = nn.ModuleList([nn.GRU(embedding_size, int(hidden_size / 2), num_layers=1, bidirectional=True, batch_first=True) if i == 0 else nn.GRU(embedding_size + hidden_size, int(hidden_size / 2), num_layers=1, bidirectional=True, batch_first=True) for i in range(num_layers)])

    def forward(self, batchsize_numseq_seqlen):
        '''
        :return: [batch_size, num_sequences, num_layers, sequence_len, hidden_size], state: [batch_size, num_sequences, num_layers, hidden_size=num_directions(2)*1/2hidden_size]
        '''
        batch_size, num_seq, seq_len=batchsize_numseq_seqlen.size()
        batchsize_numseq_seqlen=batchsize_numseq_seqlen.view(-1, seq_len)

        outputs = []
        states = []

        mask = batchsize_numseq_seqlen.ne(0).detach()
        lengths = mask.sum(dim=1).detach()

        emb = F.dropout(self.embedding(batchsize_numseq_seqlen), training=self.training)
        enc_output=emb
        for i in range(self.num_layers):
            if i>0:
                enc_output = torch.cat([enc_output, F.dropout(self.embedding(batchsize_numseq_seqlen), training=self.training)], dim=-1)
            enc_output, state = gru_forward(self.encs[i], enc_output, lengths)

            outputs.append(enc_output.unsqueeze(1))
            states.append(state.view(state.size(0), -1).unsqueeze(1))

        return torch.cat(outputs, dim=1).view(batch_size, num_seq, self.num_layers, seq_len, -1), torch.cat(states, dim=1).view(batch_size, num_seq, self.num_layers, -1)

class GRUSeqDecoder(nn.Module):
    def __init__(self, num_memories, num_layers, tgt_vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super(GRUSeqDecoder, self).__init__()
        self.tgt_vocab_size=tgt_vocab_size
        self.hidden_size=hidden_size

        if emb_matrix is None:
            self.embedding = nn.Embedding(tgt_vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)

        self.attns = nn.ModuleList([BilinearAttention(query_size=hidden_size, key_size= hidden_size, hidden_size= hidden_size) for i in range(num_memories)])

        self.dec = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, bidirectional=False, batch_first=True)

        self.gen = nn.Sequential(nn.Linear(embedding_size+hidden_size+num_memories*hidden_size, hidden_size), nn.Linear(hidden_size, self.tgt_vocab_size, bias=False), nn.Softmax(dim=-1))

        self.mix = nn.Linear(embedding_size+hidden_size+num_memories*hidden_size, num_memories + 1)

    def extend(self, dec_outputs, gen_outputs, memory_weights, source_map):
        p = F.softmax(self.mix(dec_outputs), dim=-1)

        dist1 = p[:, 0].unsqueeze(-1) * gen_outputs
        dist2 = torch.cat([p[:, i + 1].unsqueeze(-1) * memory_weights[i] for i in range(len(memory_weights))], dim=-1)
        dist2 = torch.bmm(dist2.unsqueeze(1), source_map).squeeze(1)

        return dist1 + dist2

    def forward(self, encode_memories, BOS, UNK, source_map, groundtruth_index=None, encode_weights=None, encode_masks=None, init_decoder_state=None, max_target_length=None):
        '''
        :return: [batch_size, dec_len, encode_memory_len], [batch_size, dec_len, hidden_size], [batch_size, dec_len, vocab_size]
        '''
        batch_size = source_map.size(0)
        if max_target_length is None:
            max_target_length = groundtruth_index.size(1)

        encode_weights = [w.reshape(batch_size, -1) for w in encode_weights]
        encode_memories = [encode.reshape(batch_size, -1, self.hidden_size) for encode in encode_memories]
        encode_masks = [mask.reshape(batch_size, -1) for mask in encode_masks]

        index = new_tensor([BOS] * batch_size, requires_grad=False).unsqueeze(1)

        if init_decoder_state is not None:
            dec_state = init_decoder_state.transpose(0,1)
        else:
            dec_state= None

        dec_outputs=[]
        gen_outputs=[]
        extended_gen_outputs=[]
        output_indexes=[]
        for t in range(max_target_length):
            index_emb = F.dropout(self.embedding(index), training=self.training)

            dec_output, dec_state=self.dec(index_emb, dec_state)
            dec_state=dec_state.transpose(0,1)

            attn_query=dec_output

            features=[]
            memory_attns = []
            for i in range(len(encode_memories)):
                context, context_weight, context_norm_weight = self.attns[i](attn_query, encode_memories[i], encode_memories[i], mask= encode_masks[i].unsqueeze(1))
                p = encode_weights[i].unsqueeze(1) * context_norm_weight
                p = p / (1e-8 + p.sum(dim=-1, keepdim=True))
                memory_attns.append(p.squeeze(1))
                features.append(context.squeeze(1))

            features.append(index_emb.squeeze(1))
            features.append(dec_output.squeeze(1))

            gen_output=self.gen(torch.cat(features, dim=-1))

            extended_gen_output = self.extend(torch.cat(features, dim=-1), gen_output, memory_attns, source_map)

            if self.training and groundtruth_index is not None:
                index = groundtruth_index[:, t].unsqueeze(-1)
            else:
                probs, index = topk(extended_gen_output, k=1)
                output_indexes.append(index)

            dec_outputs.append(dec_output.unsqueeze(1))
            gen_outputs.append(gen_output.unsqueeze(1))
            extended_gen_outputs.append(extended_gen_output.unsqueeze(1))

        if groundtruth_index is not None:
            output_indexes=groundtruth_index
        else:
            output_indexes=torch.cat(output_indexes, dim=1)
        return torch.cat(dec_outputs, dim=1), torch.cat(gen_outputs, dim=1), torch.cat(extended_gen_outputs, dim=1), output_indexes