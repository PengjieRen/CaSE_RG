from common.PositionalEmbedding import *
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
from common.TransformerDecoder import *
from common.TransformerEncoder import *
from common.Utils import *
from common.Interaction import *
from common.TransformerBlock import *
from common.BilinearAttention import *
from common.TransformerSeqEncoderDecoder import *

class CaSETransformerSeqDecoder(nn.Module):
    def __init__(self, num_memories, num_layers, nhead, tgt_vocab_size, hidden_size, emb_matrix=None):
        super(CaSETransformerSeqDecoder, self).__init__()
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
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)

        self.attns = nn.ModuleList([BilinearAttention(hidden_size+hidden_size, hidden_size, hidden_size) for i in range(num_memories)])

        self.gen = nn.Sequential(nn.Linear(hidden_size+hidden_size+hidden_size, hidden_size), nn.Dropout(0.1), nn.Linear(hidden_size, self.tgt_vocab_size, bias=False), nn.Softmax(dim=-1))

        self.mix = nn.Linear(3*hidden_size, num_memories+1)

    def extend(self, dec_outputs, gen_outputs, memory_weights, source_map):
        p = F.softmax(self.mix(dec_outputs), dim=-1)

        dist1=p[:, :, 0].unsqueeze(-1)*gen_outputs
        dist2=torch.cat([p[:, :, i+1].unsqueeze(-1)*memory_weights[i] for i in range(len(memory_weights))], dim=-1)
        dist2=torch.bmm(dist2, source_map)

        if self.training:
            return (dist1, dist2)
        else:
            return dist1 + dist2

    def forward(self, encode_memories, BOS, UNK, source_map, groundtruth_index=None, additional_decoder_feature=None, encode_weights=None, encode_masks=None, init_decoder_state=None, max_target_length=None):
        '''
        :return: [batch_size, dec_len, encode_memory_len], [batch_size, dec_len, hidden_size], [batch_size, dec_len, vocab_size]
        '''
        batch_size = source_map.size(0)

        encode_weights = [w.reshape(batch_size, -1) for w in encode_weights]
        encode_memories = [encode.reshape(batch_size, -1, self.hidden_size) for encode in encode_memories]
        encode_masks = [mask.reshape(batch_size, -1) for mask in encode_masks]

        if max_target_length is None:
            max_target_length = groundtruth_index.size(1)

        bos = new_tensor([BOS] * batch_size, requires_grad=False).unsqueeze(1)

        if self.training and groundtruth_index is not None:
            dec_input_index = torch.cat([bos, groundtruth_index[:, :-1]], dim=-1)
            dec_input = self.embedding(dec_input_index)

            additional_decoder_feature = F.dropout(self.norm2(additional_decoder_feature).unsqueeze(1).expand(-1, dec_input.size(1), -1), p=0.1, training=self.training)

            dec_outputs=dec_input.transpose(0, 1)
            memory_attns=[]
            c_m=[]
            for i in range(len(encode_memories)):
                dec_outputs, _, _ = self.decs[i](dec_outputs, encode_memories[i].transpose(0, 1),
                                          tgt_mask=generate_square_subsequent_mask(dec_outputs.size(0)),
                                          tgt_key_padding_mask=~dec_input_index.ne(0),
                                          memory_key_padding_mask=~encode_masks[i])
                m_i, _, m_i_weights=self.attns[i](torch.cat([dec_outputs.transpose(0, 1), additional_decoder_feature], dim=-1), encode_memories[i], encode_memories[i], mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(), encode_masks[i].unsqueeze(1).float()).bool())
                c_m.append(m_i)
                p = encode_weights[i].unsqueeze(1) * m_i_weights
                p = p / (1e-8+p.sum(dim=-1, keepdim=True))
                memory_attns.append(p)
            dec_outputs = self.norm1(dec_outputs).transpose(0, 1)

            gen_outputs = self.gen(torch.cat([dec_input, dec_outputs, additional_decoder_feature], dim=-1))

            extended_gen_outputs = self.extend(torch.cat([dec_outputs]+c_m, dim=-1), gen_outputs, memory_attns, source_map)

            output_indexes=groundtruth_index
        elif not self.training:
            input_indexes=[]
            output_indexes=[]
            for t in range(max_target_length):
                dec_input_index=torch.cat([bos] + input_indexes, dim=-1)
                dec_input = self.embedding(dec_input_index)

                additional_decoder_feature_ = F.dropout(self.norm2(additional_decoder_feature).unsqueeze(1).expand(-1, dec_input.size(1), -1), p=0.1, training=self.training)

                dec_outputs = dec_input.transpose(0, 1)
                memory_attns = []
                c_m=[]
                for i in range(len(encode_memories)):
                    dec_outputs, _, _ = self.decs[i](dec_outputs, encode_memories[i].transpose(0, 1),
                                                     tgt_mask=generate_square_subsequent_mask(dec_outputs.size(0)),
                                                     tgt_key_padding_mask=~dec_input_index.ne(0),
                                                     memory_key_padding_mask=~encode_masks[i])
                    m_i, _, m_i_weights = self.attns[i](torch.cat([dec_outputs.transpose(0, 1), additional_decoder_feature_], dim=-1), encode_memories[i], encode_memories[i], mask=torch.bmm(dec_input_index.ne(0).unsqueeze(-1).float(), encode_masks[i].unsqueeze(1).float()).bool())
                    c_m.append(m_i)
                    p = encode_weights[i].unsqueeze(1) * m_i_weights
                    p = p / (1e-8+p.sum(dim=-1, keepdim=True))
                    memory_attns.append(p)
                dec_outputs = self.norm1(dec_outputs).transpose(0, 1)

                gen_outputs = self.gen(torch.cat([dec_input, dec_outputs, additional_decoder_feature_], dim=-1))

                extended_gen_outputs = self.extend(torch.cat([dec_outputs] + c_m, dim=-1), gen_outputs, memory_attns, source_map)

                probs, indices = topk(extended_gen_outputs[:, -1], k=1)

                input_indexes.append(indices)
                output_indexes.append(indices)
            output_indexes=torch.cat(output_indexes, dim=-1)

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indexes

class RelevantPassageSelection(nn.Module):
    def __init__(self, hidden_size, num_heads, query_encoder, passage_encoder):
        super(RelevantPassageSelection, self).__init__()

        self.hidden_size = hidden_size
        self.query_encoder = query_encoder
        self.passage_encoder = passage_encoder
        self.num_heads = num_heads

        self.interaction=Interaction(hidden_size)
        self.query_blocks = nn.ModuleList([TransformerBlock(num_heads, 5*hidden_size, hidden_size)]+[TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(2)])
        self.passage_blocks = nn.ModuleList([TransformerBlock(num_heads, 5*hidden_size, hidden_size)]+[TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(4)])
        self.scorer = nn.Linear(hidden_size, 1)

    def action(self, query, passage, encode_query, encode_passage):
        '''
        :return: [batch_size, num_passage], [batch_size, num_query=1, seq_len_q, hidden_size], [batch_size, num_passage, seq_len_p, hidden_size]
        '''

        encode_query = encode_query[0][:, :, -1]
        encode_passage = encode_passage[0][:, :, -1]

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

        return passage_score, (query_reps, query_reps[:, :, 0]), (passage_reps, passage_reps[:, :, 0])

class SupportingTokenIdentification(nn.Module):
    def __init__(self, max_span_size, hidden_size, num_heads, query_encoder, passage_encoder, passage_selection):
        super(SupportingTokenIdentification, self).__init__()
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.query_encoder=query_encoder
        self.passage_encoder=passage_encoder
        self.max_span_size=max_span_size

        self.passage_selection=passage_selection

        self.interaction = Interaction(hidden_size)
        self.query_blocks = nn.ModuleList([TransformerBlock(num_heads, 5 * hidden_size, hidden_size)] + [TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(1)])
        self.passage_blocks = nn.ModuleList([TransformerBlock(num_heads, 5 * hidden_size, hidden_size)] + [TransformerBlock(num_heads, hidden_size, hidden_size) for i in range(2)])
        self.norm1 = LayerNorm(hidden_size)
        self.norm2 = LayerNorm(hidden_size)
        self.scorer = nn.Linear(hidden_size, 1)

    def action(self, query, passage, encode_query, encode_passage, passage_selection_result):
        '''
        return: [batch_size, hidden_size] [batch_size, num_span]
        '''
        passage_mask = passage.ne(0)
        query_mask = query.ne(0)

        passage_score, query_rep, passage_rep = passage_selection_result

        G_p_q, G_q_p = self.interaction(query_rep[0], passage_rep[0], query_mask, passage_mask)

        query_reps = G_p_q
        for i in range(len(self.query_blocks)):
            query_reps = self.query_blocks[i](query_reps, query_mask)
        passage_reps = G_q_p
        for i in range(len(self.passage_blocks)):
            passage_reps = self.passage_blocks[i](passage_reps, passage_mask)

        token_score = self.scorer(passage_reps).squeeze(-1)

        token_score = token_score.masked_fill(~passage_mask, -1e6)
        token_score = token_score.clamp(min=-1e6, max=1e6)

        # token_score = torch.sigmoid(passage_score).unsqueeze(-1)*torch.sigmoid(token_score)
        # token_score=token_score / (1e-8 + token_score.sum(dim=-1, keepdim=True))

        query_reps = self.norm1(query_rep[0] + query_reps)
        passage_reps = self.norm2(passage_rep[0] + passage_reps)

        return token_score, (query_reps, query_reps[:, :, 0]), (passage_reps, passage_reps[:, :, 0])

class ResponseGeneration(nn.Module):
    def __init__(self, BOS, UNK, vocab_size, hidden_size, num_heads, query_encoder, passage_encoder, passage_selection, span_extraction, decoder):
        super(ResponseGeneration, self).__init__()
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        self.num_heads=num_heads
        self.query_encoder=query_encoder
        self.passage_encoder=passage_encoder

        self.passage_selection=passage_selection
        self.span_extraction=span_extraction
        self.BOS=BOS
        self.UNK=UNK

        self.decoder=decoder

    def action(self, query, passage, source_map, encode_query, encode_passage, passage_selection_result, span_extraction_result, output=None, max_target_length=None):
        '''
        return: [batch_size, dec_len, context_query_passage_seq_len], [batch_size, dec_len, hidden_size], [batch_size, dec_len, vocab_size], [batch_size, dec_len, extended_vocab_size]
        '''
        batch_size=query.size(0)

        passage_score, query_rep, passage_rep= passage_selection_result
        token_score, query_rep, passage_rep = span_extraction_result

        prior_p = torch.sigmoid(passage_score).unsqueeze(-1)*torch.sigmoid(token_score)
        prior_p = prior_p.reshape(batch_size, -1)
        prior_p = prior_p / (1e-8 + prior_p.sum(dim=-1, keepdim=True))
        answer_rep = torch.bmm(prior_p.unsqueeze(1), passage_rep[0].reshape(batch_size, -1, passage_rep[0].size(-1))).squeeze(1)
        prior_p = prior_p.reshape_as(token_score)

        prior_q=new_tensor([1.]*batch_size).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, query_rep[0].size(2)).detach()

        dec_outputs, gen_outputs, extended_gen_outputs, output_indices = self.decoder(
            [query_rep[0], passage_rep[0]], self.BOS, self.UNK, source_map,
            additional_decoder_feature=answer_rep,
            groundtruth_index=output, max_target_length=max_target_length,
            encode_masks=[query.ne(0), passage.ne(0)], encode_weights = [prior_q, prior_p])

        return dec_outputs, gen_outputs, extended_gen_outputs, output_indices

class CaSE(nn.Module):
    def __init__(self, max_span_size, max_target_length, id2vocab, vocab2id, hidden_size):
        super(CaSE, self).__init__()

        self.UNK=vocab2id[UNK_WORD]
        self.max_target_length=max_target_length
        self.query_encoder=TransformerSeqEncoder(3, 8, len(vocab2id), hidden_size)
        self.passage_encoder=self.query_encoder
        self.passage_selection=RelevantPassageSelection(hidden_size, 8, self.query_encoder, self.passage_encoder)
        self.span_extraction=SupportingTokenIdentification(max_span_size, hidden_size, 8, self.query_encoder, self.passage_encoder, self.passage_selection)
        self.response_generation=ResponseGeneration(vocab2id[BOS_WORD], vocab2id[UNK_WORD], len(vocab2id), hidden_size, 8, self.query_encoder, self.passage_encoder, self.passage_selection, self.span_extraction, CaSETransformerSeqDecoder(2, 4, 8, len(vocab2id), hidden_size))
        self.id2vocab=id2vocab
        self.vocab_size=len(id2vocab)
        self.vocab2id=vocab2id

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def do_train(self, data):
        losses=[]
        encode_query, encode_passage = self.query_encoder(data['query']), self.passage_encoder(data['passage'])

        passage_selection_result = self.passage_selection.action(data['query'], data['passage'],
                                                                 encode_query=encode_query,
                                                                 encode_passage=encode_passage)
        # loss_ps = F.cross_entropy(passage_selection_result,  data['passage_label']).unsqueeze(0)
        label = torch.zeros_like(passage_selection_result[0]).scatter_(1, data['passage_label'].unsqueeze(-1), 1).detach()
        # loss_ps = F.binary_cross_entropy(torch.sigmoid(passage_selection_result[0]), label).unsqueeze(0)
        loss_ps = F.binary_cross_entropy_with_logits(passage_selection_result[0], label).unsqueeze(0)
        losses.append(loss_ps)

        span_extraction_result = self.span_extraction.action(data['query'], data['passage'],
                                                             encode_query=encode_query,
                                                             encode_passage=encode_passage,
                                                             passage_selection_result=passage_selection_result)
        loss_se = F.binary_cross_entropy_with_logits(span_extraction_result[0], data['token_label'].detach(), reduction='none')
        mask = data['passage'].ne(0).float().detach()
        loss_se =  (mask * loss_se * data['token_weight'].detach()).sum()/mask.sum().detach()
        losses.append(loss_se)

        response_generation_result = self.response_generation.action(data['query'], data['passage'], data['source_map'],
                                                 encode_query=encode_query,
                                                 encode_passage=encode_passage,
                                                 passage_selection_result=passage_selection_result,
                                                 span_extraction_result=span_extraction_result, output=data['response'])
        # loss_rg = F.nll_loss((response_generation_result[2]+1e-8).log().reshape(-1, response_generation_result[2].size(-1)), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
        # losses.append(loss_rg)
        dist1, dist2 = response_generation_result[2]
        # loss_rg_1 = F.nll_loss((dist1 + 1e-8).log().reshape(-1, dist1.size(-1)), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
        # loss_rg_2 = F.nll_loss(((1 - dist1).detach() * ((dist2 + 1e-8).log())).reshape(-1, dist2.size(-1)), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
        dist=dist1+dist2
        loss_rg_3 = F.nll_loss((dist + 1e-8).log().reshape(-1, dist.size(-1)), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
        # losses.append(loss_rg_1)
        # losses.append(loss_rg_2)
        losses.append(loss_rg_3)

        return losses

    def do_test(self, data):
        encode_query, encode_passage = self.query_encoder(data['query']), self.passage_encoder(data['passage'])

        passage_selection_result = self.passage_selection.action(data['query'], data['passage'],
                                                                 encode_query=encode_query,
                                                                 encode_passage=encode_passage)

        span_extraction_result = self.span_extraction.action(data['query'], data['passage'],
                                                             encode_query=encode_query,
                                                             encode_passage=encode_passage,
                                                             passage_selection_result=passage_selection_result)

        response_generation_result = self.response_generation.action(data['query'], data['passage'], data['source_map'],
                                                 encode_query=encode_query,
                                                 encode_passage=encode_passage,
                                                 passage_selection_result=passage_selection_result,
                                                 span_extraction_result=span_extraction_result, output=None, max_target_length=self.max_target_length)

        return {'answer': response_generation_result[3], 'rank':passage_selection_result[0]}

    def forward(self, data, method='mle_train'):
        if 'source_map' in data:
            data['source_map'] = build_map(data['source_map'], max=self.vocab_size)
        if method == 'train':
            return self.do_train(data)
        elif method == 'test':
            return self.do_test(data)