import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
import torch.nn.functional as F
from common.Interaction import *
from common.TransformerBlock import *
from common.PositionalEmbedding import *
from common.TransformerDecoder import *
from common.BilinearAttention import *
from common.Utils import *
from common.TransformerSeqEncoderDecoder import *

class MasqueTransformerSeqDecoder(nn.Module):
    def __init__(self, num_memories, num_layers, nhead, tgt_vocab_size, hidden_size, emb_matrix=None):
        super(MasqueTransformerSeqDecoder, self).__init__()
        self.tgt_vocab_size=tgt_vocab_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size

        if emb_matrix is None:
            self.embedding = nn.Sequential(nn.Embedding(tgt_vocab_size, hidden_size, padding_idx=0),
                                           PositionalEmbedding(hidden_size, dropout=0.1, max_len=1000))
        else:
            self.embedding = nn.Sequential(create_emb_layer(emb_matrix),
                                           PositionalEmbedding(hidden_size, dropout=0.1, max_len=1000))

        decoder_layer = TransformerDecoderLayer(hidden_size, nhead=nhead, dim_feedforward=hidden_size, dropout=0.1, activation='gelu')
        self.decs = nn.ModuleList([TransformerDecoder(decoder_layer, num_layers=num_layers, norm=None) for i in range(num_memories)])
        self.norm = LayerNorm(hidden_size)

        self.attns = nn.ModuleList([BilinearAttention(hidden_size, hidden_size, hidden_size) for i in range(num_memories)])

        self.gen = nn.Sequential(nn.Linear(hidden_size+hidden_size, hidden_size), nn.Linear(hidden_size, self.tgt_vocab_size, bias=False), nn.Softmax(dim=-1))

        self.mix = nn.Linear(3*hidden_size, num_memories+1)

    def extend(self, dec_outputs, gen_outputs, memory_weights, source_map):
        p = F.softmax(self.mix(dec_outputs), dim=-1)

        dist1=p[:, :, 0].unsqueeze(-1)*gen_outputs
        dist2=torch.cat([p[:, :, i+1].unsqueeze(-1)*memory_weights[i] for i in range(len(memory_weights))], dim=-1)
        dist2=torch.bmm(dist2, source_map)

        return dist1+dist2

    def forward(self, encode_memories, BOS, UNK, source_map, encode_masks=None, encode_weights=None, groundtruth_index=None, init_decoder_state=None, max_target_length=None):
        batch_size = source_map.size(0)

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
                                          tgt_mask=generate_square_subsequent_mask(dec_outputs.size(0)),
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

            extended_gen_outputs = self.extend(torch.cat([dec_outputs]+c_m, dim=-1), gen_outputs, memory_attns, source_map)

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
                                                     tgt_mask=generate_square_subsequent_mask(dec_outputs.size(0)),
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

                extended_gen_outputs = self.extend(torch.cat([dec_outputs] + c_m, dim=-1), gen_outputs, memory_attns, source_map)

                probs, indices = topk(extended_gen_outputs[:, -1], k=1)

                input_indexes.append(indices)
                output_indexes.append(indices)
            output_indexes=torch.cat(output_indexes, dim=-1)

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indexes

class PassageSelection(nn.Module):
    def __init__(self, hidden_size, num_heads, query_encoder, passage_encoder):
        super(PassageSelection, self).__init__()

        self.hidden_size = hidden_size
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.num_heads = num_heads

        self.interaction=Interaction(hidden_size)
        self.query_blocks = nn.ModuleList([TransformerBlock(num_heads, 5*hidden_size, hidden_size)]+[TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(2)])
        self.passage_blocks = nn.ModuleList([TransformerBlock(num_heads, 5*hidden_size, hidden_size)]+[TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(4)])
        self.scorer = nn.Linear(hidden_size, 1)

    def action(self, query, passage, encode_query=None, encode_passage=None):
        '''
        :return: [batch_size, num_passage], [batch_size, num_query=1, seq_len_q, hidden_size], [batch_size, num_passage, seq_len_p, hidden_size]
        '''

        if encode_query is None:
            encode_query = self.query_encoder(query)[0][:,:,-1]
        if encode_passage is None:
            encode_passage = self.passage_encoder(passage)[0][:,:,-1]

        passage_mask=passage.ne(0)
        query_mask=query.ne(0)

        G_p_q, G_q_p=self.interaction(encode_query, encode_passage, query_mask, passage_mask)

        query_reps=G_p_q
        for i in range(len(self.query_blocks)):
            query_reps=self.query_blocks[i](query_reps, query_mask)
        passage_reps=G_q_p
        for i in range(len(self.passage_blocks)):
            passage_reps = self.passage_blocks[i](passage_reps, passage_mask)

        passage_score = self.scorer(passage_reps[:, :, 0]).squeeze(-1)  # passage_reps[:,:,0]:[batch_size, num_sequences, hidden_size], 0->[CLS]

        return passage_score, query_reps, passage_reps

class ResponseGeneration(nn.Module):
    def __init__(self, BOS, UNK, vocab_size, hidden_size, num_heads, query_encoder, passage_encoder, passage_selection, decoder):
        super(ResponseGeneration, self).__init__()
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_heads=num_heads
        self.query_encoder=query_encoder
        self.passage_encoder=passage_encoder

        self.passage_selection=passage_selection
        self.BOS=BOS
        self.UNK=UNK

        self.decoder=decoder

    def action(self, query, passage, source_map, encode_query=None, encode_passage=None, passage_selection_result=None, output=None, max_target_length=None):
        '''
        return: [batch_size, dec_len, context_query_passage_seq_len], [batch_size, dec_len, hidden_size], [batch_size, dec_len, vocab_size], [batch_size, dec_len, extended_vocab_size]
        '''
        if encode_query is None:
            encode_query = self.query_encoder(query)[0][:, :, -1]
        if encode_passage is None:
            encode_passage = self.passage_encoder(passage)[0][:, :, -1]
        if passage_selection_result is None:
            passage_selection_result = self.passage_selection.action(query, passage, encode_query=encode_query, encode_passage=encode_passage)
        passage_score, query_representation, passage_representation = passage_selection_result

        batch_size=query.size(0)

        prior_q = new_tensor([1.] * batch_size).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, query_representation.size(2)).detach()
        prior_p = torch.sigmoid(passage_score).unsqueeze(-1).expand(-1, -1, passage_representation.size(2))

        # print('pp', torch.sigmoid(passage_score))

        dec_outputs, gen_outputs, extended_gen_outputs, output_indices = self.decoder(
            [query_representation, passage_representation], self.BOS, self.UNK, source_map,
            groundtruth_index=output, max_target_length=max_target_length,
            encode_masks=[query.ne(0), passage.ne(0)], encode_weights = [prior_q, prior_p])

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indices

class Masque(nn.Module):
    def __init__(self, max_target_length, id2vocab, vocab2id, hidden_size):
        super(Masque, self).__init__()

        self.UNK=vocab2id[UNK_WORD]
        self.max_target_length=max_target_length

        self.query_encoder=TransformerSeqEncoder(3, 8, len(vocab2id), hidden_size)
        self.passage_encoder=self.query_encoder
        self.passage_selection=PassageSelection(hidden_size, 8, self.query_encoder, self.passage_encoder)
        self.response_generation=ResponseGeneration(vocab2id[BOS_WORD], vocab2id[UNK_WORD], len(vocab2id), hidden_size, 8, self.query_encoder, self.passage_encoder, self.passage_selection, MasqueTransformerSeqDecoder(2, 4, 8, len(vocab2id), hidden_size))
        self.id2vocab=id2vocab
        self.vocab_size=len(id2vocab)
        self.vocab2id=vocab2id

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def do_train(self, data):
        losses=[]
        encode_query, encode_passage = self.query_encoder(
            data['query']), self.passage_encoder(data['passage'])
        encode_query, encode_passage=encode_query[0][:, :, -1], encode_passage[0][:, :, -1]

        passage_selection_result = self.passage_selection.action(data['query'], data['passage'],
                                                                 encode_query=encode_query,
                                                                 encode_passage=encode_passage)
        label = torch.zeros_like(passage_selection_result[0]).scatter_(1, data['passage_label'].unsqueeze(-1), 1)
        # loss_ps = F.binary_cross_entropy(torch.sigmoid(passage_selection_result[0]), label).unsqueeze(0)
        loss_ps = F.binary_cross_entropy_with_logits(passage_selection_result[0], label).unsqueeze(0)
        losses.append(0.25*loss_ps)

        response_generation_result = self.response_generation.action(data['query'], data['passage'], data['source_map'],
                                                 encode_query=encode_query,
                                                 encode_passage=encode_passage,
                                                 passage_selection_result=passage_selection_result,
                                                 output=data['response'])
        loss_rg = F.nll_loss((response_generation_result[2]+1e-8).log().reshape(-1, response_generation_result[2].size(-1)), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
        losses.append(loss_rg)

        return losses

    def do_ps_train(self, data):
        losses=[]
        encode_query, encode_passage = self.query_encoder(
            data['query']), self.passage_encoder(data['passage'])
        encode_query, encode_passage=encode_query[0][:, :, -1], encode_passage[0][:, :, -1]

        passage_selection_result = self.passage_selection.action(data['query'], data['passage'],
                                                                 encode_query=encode_query,
                                                                 encode_passage=encode_passage)
        label = torch.zeros_like(passage_selection_result[0]).scatter_(1, data['passage_label'].unsqueeze(-1), 1)
        # loss_ps = F.binary_cross_entropy(torch.sigmoid(passage_selection_result[0]), label).unsqueeze(0)
        loss_ps = F.binary_cross_entropy_with_logits(passage_selection_result[0], label).unsqueeze(0)
        losses.append(loss_ps)

        return losses

    def do_test(self, data):
        encode_query, encode_passage = self.query_encoder(
            data['query']), self.passage_encoder(data['passage'])
        encode_query, encode_passage = encode_query[0][:, :, -1], encode_passage[0][:, :, -1]

        passage_selection_result = self.passage_selection.action(data['query'], data['passage'],
                                                                 encode_query=encode_query,
                                                                 encode_passage=encode_passage)

        response_generation_result = self.response_generation.action(data['query'], data['passage'], data['source_map'],
                                                 encode_query=encode_query,
                                                 encode_passage=encode_passage,
                                                 passage_selection_result=passage_selection_result,
                                                 output=None, max_target_length=self.max_target_length)
        # loss_rg = mix_dist_criterion(response_generation_result[2], data['response'], data['dyn_response'], self.UNK, self.vocab_size).unsqueeze(0)
        # print('test: ',loss_rg)

        return {'answer':response_generation_result[3], 'rank':passage_selection_result[0]}

    def forward(self, data, method='mle_train'):
        data['source_map'] = build_map(data['source_map'], max=self.vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'ps_train':
            return self.do_ps_train(data)
        elif method == 'test':
            return self.do_test(data)