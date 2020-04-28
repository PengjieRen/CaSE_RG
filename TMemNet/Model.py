from TMemNet.EncDecModel import *
from common.PositionalEmbedding import *
import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
from common.TransformerEncoder import *
from common.TransformerDecoder import *

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504
def neginf(dtype):
    """Return a representable finite number near -inf for a dtype."""
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(~mask, neginf(torch.float32)).masked_fill(mask, float(0.0))
    if torch.cuda.is_available():
        mask = mask.cuda()
    return mask

def universal_sentence_embedding(sentences, mask, sqrt=True):
    '''
    :param sentences: [batch_size, seq_len, hidden_size]
    :param mask: [batch_size, seq_len]
    :param sqrt:
    :return: [batch_size, hidden_size]
    '''
    # need to mask out the padded chars
    sentence_sums = torch.bmm(
        sentences.permute(0, 2, 1), mask.float().unsqueeze(-1)
    ).squeeze(-1)
    divisor = (mask.sum(dim=1).view(-1, 1).float())
    if sqrt:
        divisor = divisor.sqrt()
    sentence_sums /= divisor
    return sentence_sums

class ContextKnowledgeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super().__init__()
        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)
        self.pos_embedding = PositionalEmbedding(embedding_size)

        encoder_layers = TransformerEncoderLayer(hidden_size, nhead=8, dim_feedforward=hidden_size, dropout=0.1, activation='gelu')
        self.transformer = TransformerEncoder(encoder_layers, num_layers=8)

    def forward(self, src_tokens, know_tokens, context_mask, know_mask, cs_ids=None):
        src_tokens = self.pos_embedding(self.embedding(src_tokens)).transpose(0,1)
        know_tokens = self.pos_embedding(self.embedding(know_tokens))

        # encode the context, pretty basic
        context_encoded = self.transformer(src_tokens, src_key_padding_mask=~context_mask).transpose(0,1)

        # make all the knowledge into a 2D matrix to encode
        N, K, L, H = know_tokens.size()
        know_mask =know_mask.view(-1, L)
        know_encoded = self.transformer(know_tokens.view(-1, L, H).transpose(0, 1), src_key_padding_mask=~know_mask).transpose(0,1)

        # compute our sentence embeddings for context and knowledge
        context_use = universal_sentence_embedding(context_encoded, context_mask)
        know_use = universal_sentence_embedding(know_encoded, know_mask)

        # remash it back into the shape we need
        know_use = know_use.view(N, K, H)
        context_use /= np.sqrt(H)
        know_use /= np.sqrt(H)

        ck_attn = torch.bmm(know_use, context_use.unsqueeze(-1)).squeeze(-1)

        # if we're not given the true chosen_sentence (test time), pick our
        # best guess
        if cs_ids is None or not self.training:
            _, cs_ids = ck_attn.max(1)

        # pick the true chosen sentence. remember that TransformerEncoder outputs
        #   (batch, time, embed)
        # but because know_encoded is a flattened, it's really
        #   (N * K, T, D)
        # We need to compute the offsets of the chosen_sentences
        cs_offsets = torch.arange(N, device=cs_ids.device) * K + cs_ids

        cs_encoded = know_encoded[cs_offsets]
        # but padding is (N * K, T)
        cs_mask = know_mask[cs_offsets]

        # finally, concatenate it all
        full_enc = torch.cat([cs_encoded, context_encoded], dim=1)
        full_mask = torch.cat([cs_mask, context_mask], dim=1)

        # also return the knowledge selection mask for the loss
        return full_enc, full_mask, ck_attn

class ContextKnowledgeDecoder(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super().__init__()
        if emb_matrix is None:
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding = create_emb_layer(emb_matrix)
        self.pos_embedding = PositionalEmbedding(embedding_size)

        decoder_layer = TransformerDecoderLayer(hidden_size, nhead=8, dim_feedforward=hidden_size, dropout=0.1, activation='gelu')
        self.transformer = TransformerDecoder(decoder_layer, num_layers=8)

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        tgt = self.pos_embedding(self.embedding(tgt)).transpose(0,1)

        rs, _, _= self.transformer(tgt, memory.transpose(0,1), tgt_mask=_generate_square_subsequent_mask(tgt.size(0)), tgt_key_padding_mask=~tgt_mask, memory_key_padding_mask=~memory_mask)
        return rs.transpose(0,1)

class TMemNet(EncDecModel):
    def __init__(self,embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len=120, beam_width=1, eps=1e-10, emb_matrix=None):
        super(TMemNet, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)

        self.id2vocab=id2vocab
        self.hidden_size=hidden_size

        self.enc = ContextKnowledgeEncoder(len(vocab2id), embedding_size, hidden_size, emb_matrix=emb_matrix)
        self.dec= ContextKnowledgeDecoder(len(vocab2id), embedding_size, hidden_size, emb_matrix=emb_matrix)
        self.gen = nn.Linear(self.hidden_size, len(vocab2id))

    def encode(self, data):
        c=data['context']
        c_mask = c.ne(0).detach()
        p=data['passage']
        p_mask = p.ne(0).detach()

        memory, memory_mask, ck_attn=self.enc(c, p, c_mask, p_mask, data['label'])
        return {'memory': memory, 'memory_mask': memory_mask, 'passage_selection': ck_attn}

    def init_decoder_states(self,data, encode_output):
        return {}

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, decode_step):
        if decode_step ==0:
            current_words=previous_word.unsqueeze(1)
        else:
            current_words=torch.cat([previous_deocde_outputs['output'], previous_word.unsqueeze(1)], dim=1)
        output = self.dec(current_words, encode_outputs['memory'], tgt_mask=current_words.ne(0), memory_mask= encode_outputs['memory_mask'])
        return {'feature': output[:,-1], 'output':current_words}

    def generate(self,data, encode_outputs, decode_outputs, softmax=True):
        return self.gen(decode_outputs['feature'])

    def to_word(self, data, gen_output, k=5, sampling=False):
        if not sampling:
            return topk(gen_output, k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def do_train(self, data):
        encode_outputs=self.encode(data)
        tgt_input=torch.cat([new_tensor([self.vocab2id[BOS_WORD]] * data['response'].size(0), requires_grad=False).long().unsqueeze(1), data['response']], dim=1)
        tgt_output=torch.cat([data['response'], new_tensor([self.vocab2id[PAD_WORD]] * data['response'].size(0), requires_grad=False).long().unsqueeze(1)], dim=1)

        output = self.dec(tgt_input, encode_outputs['memory'], tgt_mask=tgt_input.ne(0), memory_mask= encode_outputs['memory_mask'])
        gen_output=self.gen(output)

        # loss_s = F.cross_entropy(encode_outputs['passage_selection'], data['label'].view(-1))
        label = torch.zeros_like(encode_outputs['passage_selection']).scatter_(1, data['label'].unsqueeze(-1), 1)
        # loss_s = F.binary_cross_entropy(torch.sigmoid(encode_outputs['passage_selection']), label)
        loss_s = F.binary_cross_entropy_with_logits(encode_outputs['passage_selection'], label)
        loss_g = F.cross_entropy(gen_output.view(-1, gen_output.size(-1)), tgt_output.view(-1), ignore_index=0)
        return 0.25*loss_s.unsqueeze(0), loss_g.unsqueeze(0)

    def do_ps_train(self, data):
        encode_outputs=self.encode(data)

        # loss_s = F.cross_entropy(encode_outputs['passage_selection'], data['label'].view(-1))
        label = torch.zeros_like(encode_outputs['passage_selection']).scatter_(1, data['label'].unsqueeze(-1), 1)
        # loss_s = F.binary_cross_entropy(torch.sigmoid(encode_outputs['passage_selection']), label)
        loss_s = F.binary_cross_entropy_with_logits(encode_outputs['passage_selection'], label)
        return loss_s.unsqueeze(0)

    def forward(self, data, method='train'):
        if method=='train':
            return self.do_train(data)
        elif method=='ps_train':
            return self.do_ps_train(data)
        elif method=='test':
            if self.beam_width==1:
                return {'answer': self.greedy(data), 'rank':self.encode(data)['passage_selection']}
            else:
                return {'answer': self.beam(data), 'rank':self.encode(data)['passage_selection']}