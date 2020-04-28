from common.PositionalEmbedding import *
from common.TransformerEncoder import *
from common.TransformerDecoder import *
from common.Utils import *
from common.BilinearAttention import *

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, neginf(torch.float32)).masked_fill(mask, float(0.0))
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask

class TransformerSeqEncoder(nn.Module):
    def __init__(self, num_layers, num_heads, src_vocab_size, hidden_size, emb_matrix=None, norm=None):
        super(TransformerSeqEncoder, self).__init__()
        self.num_layers=num_layers
        self.num_heads=num_heads

        if emb_matrix is None:
            self.embedding = nn.Sequential(nn.Embedding(src_vocab_size, hidden_size, padding_idx=0), PositionalEmbedding(hidden_size, dropout=0.1, max_len=1000))
        else:
            self.embedding = nn.Sequential(create_emb_layer(emb_matrix), PositionalEmbedding(hidden_size, dropout=0.1, max_len=1000))

        encoder_layers = TransformerEncoderLayer(hidden_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=0.1, activation='gelu')
        self.enc = TransformerEncoder(encoder_layers, num_layers=num_layers, norm=norm)

    def forward(self, batch_numseq_seqlen):
        '''
        :return: [batch_size, num_sequences, num_layers, sequence_len, hidden_size], state: [batch_size, num_sequences, num_layers, hidden_size=num_directions(2)*1/2hidden_size]
        '''
        batch_size, num_seq, seq_len=batch_numseq_seqlen.size()
        batch_numseq_seqlen=batch_numseq_seqlen.reshape(-1, seq_len)
        mask = batch_numseq_seqlen.ne(0).detach()

        emb = self.embedding(batch_numseq_seqlen)

        output=self.enc(emb.transpose(0,1), src_key_padding_mask=~mask).transpose(0,1)

        state=universal_sentence_embedding(output, mask)

        output=output.reshape(batch_size, num_seq, seq_len, -1)
        state=state.reshape(batch_size, num_seq, -1)

        return output.unsqueeze(2), state.unsqueeze(2)

class TransformerSeqDecoder(nn.Module):
    def __init__(self, num_memories, num_layers, nhead, tgt_vocab_size, hidden_size, emb_matrix=None):
        super(TransformerSeqDecoder, self).__init__()
        self.tgt_vocab_size=tgt_vocab_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size

        if emb_matrix is None:
            self.embedding = nn.Sequential(nn.Embedding(tgt_vocab_size, hidden_size, padding_idx=0), PositionalEmbedding(hidden_size, dropout=0.1, max_len=1000))
        else:
            self.embedding = nn.Sequential(create_emb_layer(emb_matrix), PositionalEmbedding(hidden_size, dropout=0.1, max_len=100))

        decoder_layer = TransformerDecoderLayer(hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=0.1, activation='gelu')
        self.decs = nn.ModuleList([TransformerDecoder(decoder_layer, num_layers=num_layers, norm=None) for i in range(num_memories)])
        self.norm = LayerNorm(hidden_size)

        self.attns = nn.ModuleList([BilinearAttention(hidden_size, hidden_size, hidden_size) for i in range(num_memories)])

        self.gen = nn.Sequential(nn.Linear(hidden_size+hidden_size, hidden_size), nn.Linear(hidden_size, self.tgt_vocab_size, bias=False), nn.Softmax(dim=-1))

        self.mix = nn.Linear(hidden_size+num_memories*hidden_size, num_memories+1)

    def extend(self, dec_outputs, gen_outputs, memory_weights, source_maps):
        p = F.softmax(self.mix(dec_outputs), dim=-1)

        dist1=p[:, :, 0].unsqueeze(-1)*gen_outputs
        dist2=torch.cat([p[:, :, i+1].unsqueeze(-1)*memory_weights[i] for i in range(len(memory_weights))], dim=-1)
        dist2=torch.bmm(dist2, torch.cat(source_maps, dim=-2))

        return dist1+dist2

    def forward(self, encode_memories, BOS, UNK, source_maps, encode_masks=None, encode_weights=None, groundtruth_index=None, init_decoder_state=None, max_target_length=None):
        batch_size = source_maps[0].size(0)

        if encode_weights is not None:
            encode_weights = [w.reshape(batch_size, -1) for w in encode_weights]
        encode_memories = [encode.reshape(batch_size, -1, self.hidden_size) for encode in encode_memories]
        encode_masks = [mask.reshape(batch_size, -1) for mask in encode_masks]

        if max_target_length is None:
            max_target_length = groundtruth_index.size(1)

        bos = new_tensor([BOS] * batch_size, requires_grad=False).unsqueeze(1)

        if self.training and groundtruth_index is not None:
            dec_input_index = torch.cat([bos, groundtruth_index[:, :-1]], dim=-1)
            dec_input = self.embedding(dec_input_index)

            dec_outputs=dec_input.transpose(0, 1)
            memory_attns=[]
            c_m=[]
            for i in range(len(encode_memories)):
                dec_outputs, _, _ = self.decs[i](dec_outputs, encode_memories[i].transpose(0, 1),
                                          tgt_mask=_generate_square_subsequent_mask(dec_outputs.size(0)),
                                          tgt_key_padding_mask=~dec_input_index.ne(0),
                                          memory_key_padding_mask=~encode_masks[i])
                m_i, _, m_i_weights=self.attns[i](dec_outputs.transpose(0, 1), encode_memories[i], encode_memories[i], mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(), encode_masks[i].unsqueeze(1).float()).bool())
                c_m.append(m_i)
                p = m_i_weights
                if encode_weights is not None:
                    p = encode_weights[i].unsqueeze(1) * p
                    p = p / (1e-8+p.sum(dim=-1, keepdim=True))
                memory_attns.append(p)
            dec_outputs = self.norm(dec_outputs).transpose(0, 1)

            gen_outputs = self.gen(torch.cat([dec_input, dec_outputs], dim=-1))

            extended_gen_outputs = self.extend(torch.cat([dec_outputs]+c_m, dim=-1), gen_outputs, memory_attns, source_maps)

            output_indexes=groundtruth_index
        elif not self.training:
            input_indexes=[]
            output_indexes=[]
            for t in range(max_target_length):
                dec_input_index=torch.cat([bos] + input_indexes, dim=-1)
                dec_input = self.embedding(dec_input_index)

                dec_outputs = dec_input.transpose(0, 1)
                memory_attns = []
                c_m=[]
                for i in range(len(encode_memories)):
                    dec_outputs, _, _ = self.decs[i](dec_outputs, encode_memories[i].transpose(0, 1),
                                                     tgt_mask=_generate_square_subsequent_mask(dec_outputs.size(0)),
                                                     tgt_key_padding_mask=~dec_input_index.ne(0),
                                                     memory_key_padding_mask=~encode_masks[i])
                    m_i, _, m_i_weights = self.attns[i](dec_outputs.transpose(0, 1), encode_memories[i], encode_memories[i], mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(), encode_masks[i].unsqueeze(1).float()).bool())
                    c_m.append(m_i)
                    p = m_i_weights
                    if encode_weights is not None:
                        p = encode_weights[i].unsqueeze(1) * p
                        p = p / (1e-8+p.sum(dim=-1, keepdim=True))
                    memory_attns.append(p)
                dec_outputs = self.norm(dec_outputs).transpose(0, 1)

                gen_outputs = self.gen(torch.cat([dec_input, dec_outputs], dim=-1))

                extended_gen_outputs = self.extend(torch.cat([dec_outputs] + c_m, dim=-1), gen_outputs, memory_attns, source_maps)

                probs, indices = topk(extended_gen_outputs[:, -1], k=1)

                input_indexes.append(indices)
                output_indexes.append(indices)
            output_indexes=torch.cat(output_indexes, dim=-1)

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indexes