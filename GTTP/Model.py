from GTTP.EncDecModel import *
from common.BilinearAttention import *
from common.Utils import *

class CopyGenerator(nn.Module):
    def __init__(self, feature_size, tgt_vocab_size, logit_scale=1, weight=None):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(feature_size, tgt_vocab_size)
        self.linear_copy = nn.Linear(feature_size, 1)
        self.logit_scale = logit_scale
        if weight is not None:
            self.linear.weight = weight

    def forward(self, feature, attention, src_map):
        """
        attention: [batch, source_len] batch can be batch*target_len
        src_map: [batch, source_len, copy_vocab_size] value: 0 or 1
        """
        # CHECKS
        batch, _ = feature.size()
        batch, slen = attention.size()
        batch,slen,cvocab = src_map.size()

        # Original probabilities.
        logits = self.linear(feature) * self.logit_scale
        logits[:, 0] = -float('inf')
        logits = F.softmax(logits, dim=1)

        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(feature))
        # Probibility of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(logits,  1 - p_copy.expand_as(logits))
        # mul_attn = torch.mul(attention, p_copy.expand_as(attention))
        # copy_prob = torch.bmm(mul_attn.view(-1, batch, slen)
        #                       .transpose(0, 1),
        #                       src_map.float()).squeeze(1)

        copy_prob = torch.bmm(attention.view(-1, batch, slen)
                              .transpose(0, 1),
                              src_map.float()).squeeze(1)
        copy_prob = torch.mul(copy_prob, p_copy.expand_as(copy_prob))

        return out_prob+copy_prob
        # return out_prob, copy_prob

# class CopyCriterion(object):
#     def __init__(self, tgt_vocab_size, force_copy=False, eps=1e-20):
#         self.force_copy = force_copy
#         self.eps = eps
#         self.offset = tgt_vocab_size
#
#     def __call__(self, gen_output, tgt, tgt_copy, UNK, reduction='mean'):
#         copy_unk = tgt_copy.eq(UNK).float()
#         copy_not_unk = tgt_copy.ne(UNK).float()
#         target_unk = tgt.eq(UNK).float()
#         target_not_unk = tgt.ne(UNK).float()
#         target_not_pad=tgt.ne(0).float()
#
#         # Copy probability of tokens in source
#         if len(gen_output.size())>2:
#             gen_output = gen_output.view(-1, gen_output.size(-1))
#         copy_p = gen_output.gather(1, tgt_copy.view(-1, 1) + self.offset).view(-1)
#         # Set scores for unk to 0 and add eps
#         copy_p = copy_p.mul(copy_not_unk.view(-1)) + self.eps
#         # Get scores for tokens in target
#         tgt_p = gen_output.gather(1, tgt.view(-1, 1)).view(-1)
#
#         # Regular prob (no unks and unks that can't be copied)
#         if not self.force_copy:
#             # Add score for non-unks in target
#             p = copy_p + tgt_p.mul(target_not_unk.view(-1))
#             # Add score for when word is unk in both align and tgt
#             p = p + tgt_p.mul(copy_unk.view(-1)).mul(target_unk.view(-1))
#         else:
#             # Forced copy. Add only probability for not-copied tokens
#             p = copy_p + tgt_p.mul(copy_unk.view(-1))
#
#         p = p.log()
#
#         # Drop padding.
#         loss = -p.mul(target_not_pad.view(-1))
#         if reduction=='mean':
#             return loss.sum()/target_not_pad.sum()
#         elif reduction=='none':
#             return loss.view(tgt.size())

class BBCDecoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size, emb_matrix=None, num_layers=4, dropout=0.5):
        super(BBCDecoder, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.embedding_size=embedding_size
        self.vocab_size = vocab_size

        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)
        self.embedding_dropout = nn.Dropout(dropout)

        self.src_attn = BilinearAttention(
            query_size=hidden_size, key_size=2*hidden_size, hidden_size=hidden_size
        )
        self.bg_attn = BilinearAttention(
            query_size=hidden_size, key_size=2 * hidden_size, hidden_size=hidden_size
        )

        self.gru = nn.GRU(2*hidden_size+2*hidden_size+embedding_size, hidden_size, bidirectional=False, num_layers=num_layers)

        self.readout = nn.Linear(embedding_size + hidden_size + 2*hidden_size+ 2*hidden_size, hidden_size)

    def forward(self, tgt, state, src_output, bg_output, src_mask=None, bg_mask=None):
        embedded = self.embedding(tgt)
        embedded = self.embedding_dropout(embedded)

        src_context,_, src_attn=self.src_attn(state[:,-1].unsqueeze(1), src_output, src_output, mask=src_mask.unsqueeze(1))
        src_context=src_context.squeeze(1)
        src_attn = src_attn.squeeze(1)
        bg_context,_, bg_attn = self.bg_attn(state[:, -1].unsqueeze(1), bg_output, bg_output, mask=bg_mask.unsqueeze(1))
        bg_context = bg_context.squeeze(1)
        bg_attn = bg_attn.squeeze(1)

        gru_input = torch.cat((embedded, src_context, bg_context), dim=1)
        gru_output, gru_state=self.gru(gru_input.unsqueeze(0), state.transpose(0,1))
        gru_state=gru_state.transpose(0,1)

        concat_output = torch.cat((embedded, gru_state[:,-1], src_context, bg_context), dim=1)

        feature_output=self.readout(concat_output)
        return feature_output, [gru_state], [src_attn, bg_attn], bg_context

class GTTP(EncDecModel):
    def __init__(self,embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=120, beam_width=1, eps=1e-10, emb_matrix=None):
        super(GTTP, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)

        self.id2vocab=id2vocab
        self.hidden_size=hidden_size

        if emb_matrix is None:
            self.c_embedding = nn.Embedding(len(vocab2id), embedding_size, padding_idx=0)
        else:
            self.c_embedding = create_emb_layer(emb_matrix)
        self.b_embedding = self.c_embedding
        self.c_embedding_dropout = nn.Dropout(0.5)
        self.b_embedding_dropout = nn.Dropout(0.5)

        self.c_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.b_enc = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True)

        self.enc2dec = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.dec = BBCDecoder(embedding_size, hidden_size, len(vocab2id), num_layers=1, dropout=0.5, emb_matrix=emb_matrix)
        self.gen = CopyGenerator(feature_size=self.hidden_size, tgt_vocab_size=len(vocab2id))
        # self.criterion = CopyCriterion(len(vocab2id), force_copy=False, eps=eps)

    def encode(self, data):
        c_mask = data['context'].ne(0).detach()
        b_mask = data['background'].ne(0).detach()

        c_words = self.c_embedding_dropout(self.c_embedding(data['context']))
        b_words=self.b_embedding_dropout(self.b_embedding(data['background']))

        c_lengths=c_mask.sum(dim=1).detach()
        b_lengths = b_mask.sum(dim=1).detach()
        c_enc_output, c_state = gru_forward(self.c_enc, c_words, c_lengths)
        b_enc_output, b_state = gru_forward(self.b_enc, b_words, b_lengths)

        return c_enc_output, c_state, b_enc_output, b_state

    def init_decoder_states(self,data, encode_output):
        c_state=encode_output[1]
        batch_size=encode_output[0].size(0)

        return self.enc2dec(c_state.contiguous().view(batch_size,-1)).view(batch_size,1,-1)

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs):
        c_mask=data['context'].ne(0)
        b_mask = data['background'].ne(0)
        feature_output, [gru_state], [src_attn, bg_attn], bg_context = self.dec(previous_word, previous_deocde_outputs['state'], encode_outputs[0], encode_outputs[2], src_mask=c_mask, bg_mask=b_mask)
        return {'state': gru_state, 'feature': feature_output, 'bg_attn':bg_attn}


    def generate(self,data, encode_outputs, decode_outputs, softmax=True):
        return self.gen(decode_outputs['feature'], decode_outputs['bg_attn'], data['background_map'])

    def generation_to_decoder_input(self, data, indices):
        return indices

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return topk(gen_output, k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def mle_train(self, data):
        encode_output, init_decoder_state, all_decode_output, all_gen_output = decode_to_end(self, data, self.vocab2id, tgt=data['response'])
        gen_output=torch.cat([p.unsqueeze(1) for p in all_gen_output], dim=1)
        loss = F.nll_loss((gen_output.reshape(-1, gen_output.size(-1))+1e-8).log(), data['response'].reshape(-1), ignore_index=0)
        return loss.unsqueeze(0)

    def forward(self, data, method='train'):
        data['background_map'] = build_map(data['background_map'], max=len(self.vocab2id))
        if method=='train':
            return self.mle_train(data)
        elif method=='test':
            if self.beam_width==1:
                return {'answer':self.greedy(data)}
            else:
                return {'answer':self.beam(data)}