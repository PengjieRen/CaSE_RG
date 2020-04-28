from GLKS.EncDecModel import *
from common.BilinearAttention import *
from torch.distributions.categorical import Categorical
from common.Highway import *

class GenEncoder(nn.Module):
    def __init__(self, n, src_vocab_size, embedding_size, hidden_size, emb_matrix=None):
        super(GenEncoder, self).__init__()
        self.n=n

        if emb_matrix is None:
            self.c_embedding = nn.ModuleList([nn.Embedding(src_vocab_size, embedding_size, padding_idx=0) for i in range(n)])
        else:
            self.c_embedding = nn.ModuleList([create_emb_layer(emb_matrix) for i in range(n)])
        self.c_encs = nn.ModuleList([nn.GRU(embedding_size, int(hidden_size/2), num_layers=1, bidirectional=True, batch_first=True) if i==0 else nn.GRU(embedding_size+hidden_size, int(hidden_size/2), num_layers=1, bidirectional=True, batch_first=True) for i in range(n)])

    def forward(self, c):
        c_outputs = []
        c_states = []

        c_mask = c.ne(0).detach()
        c_lengths = c_mask.sum(dim=1).detach()

        c_emb = F.dropout(self.c_embedding[0](c), training=self.training)
        c_enc_output=c_emb
        for i in range(self.n):
            if i>0:
                c_enc_output = torch.cat([c_enc_output, F.dropout(self.c_embedding[i](c), training=self.training)], dim=-1)
            c_enc_output, c_state = gru_forward(self.c_encs[i], c_enc_output, c_lengths)

            c_outputs.append(c_enc_output.unsqueeze(1))
            c_states.append(c_state.view(c_state.size(0), -1).unsqueeze(1))

        return torch.cat(c_outputs, dim=1), torch.cat(c_states, dim=1)

class KnowledgeSelector(nn.Module):
    def __init__(self, hidden_size, min_window_size=5, n_windows=4):
        super(KnowledgeSelector, self).__init__()
        self.min_window_size=min_window_size
        self.n_windows=n_windows

        self.b_highway = Highway(hidden_size * 2, hidden_size*2, num_layers=2)
        self.c_highway = Highway(hidden_size * 2, hidden_size*2, num_layers=2)
        self.match_attn = BilinearAttention(query_size=hidden_size*2, key_size=hidden_size*2, hidden_size=hidden_size*2)
        self.area_attn = BilinearAttention(query_size=hidden_size, key_size=hidden_size, hidden_size=hidden_size)

    def match(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        b_enc_output = self.b_highway(torch.cat([b_enc_output, c_state.expand(-1, b_enc_output.size(1), -1)], dim=-1))
        c_enc_output = self.c_highway(torch.cat([c_enc_output, c_state.expand(-1, c_enc_output.size(1), -1)], dim=-1))

        matching = self.match_attn.matching(b_enc_output, c_enc_output)

        matching = matching.masked_fill(~c_mask.unsqueeze(1), -float('inf'))
        matching = matching.masked_fill(~b_mask.unsqueeze(2), 0)

        score = matching.max(dim=-1)[0]

        return score

    def segments(self, b_enc_output, b_score, c_state):
        window_size = self.min_window_size
        bs = list()
        ss = list()
        for i in range(self.n_windows):
            b = b_enc_output.unfold(1, window_size, self.min_window_size)
            b = b.transpose(2, 3).contiguous()
            b = self.area_attn(c_state.unsqueeze(1), b, b)[0].squeeze(2)
            bs.append(b)

            s = b_score.unfold(1, window_size, self.min_window_size)
            s = s.sum(dim=-1)
            ss.append(s)

            window_size += self.min_window_size
        return torch.cat(bs, dim=1), torch.cat(ss, dim=1)

    def forward(self, b_enc_output, c_enc_output, c_state, b_mask, c_mask):
        b_score=self.match(b_enc_output, c_enc_output, c_state, b_mask, c_mask)
        segments, s_score=self.segments(b_enc_output, b_score, c_state)

        s_score = F.softmax(s_score, dim=-1)

        segments = torch.bmm(s_score.unsqueeze(1), segments)

        return segments, s_score, b_score

class CopyGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(CopyGenerator, self).__init__()

        self.b_attn = BilinearAttention(query_size=embedding_size+hidden_size * 2, key_size=hidden_size, hidden_size=hidden_size)

    def forward(self, p, word, state, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        p = self.b_attn.score(torch.cat([word, state, segment], dim=-1), b_enc_output, mask=b_mask.unsqueeze(1))[1].squeeze(1)
        return p

class VocabGenerator(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab_size):
        super(VocabGenerator, self).__init__()

        self.c_attn = BilinearAttention(query_size=embedding_size+hidden_size*2, key_size=hidden_size, hidden_size=hidden_size)
        self.b_attn = BilinearAttention(query_size=embedding_size+hidden_size*2, key_size=hidden_size, hidden_size=hidden_size)

        self.readout = nn.Linear(embedding_size+4*hidden_size, hidden_size)
        self.generator = nn.Linear(hidden_size, vocab_size)

    def forward(self, p, word, state, segment, b_enc_output, c_enc_output, b_mask, c_mask):
        c_output, _, _=self.c_attn(torch.cat([word, state, segment], dim=-1), c_enc_output, c_enc_output, mask=c_mask.unsqueeze(1))
        c_output = c_output.squeeze(1)

        b_output, _, _=self.b_attn(torch.cat([word, state, segment], dim=-1), b_enc_output, b_enc_output, mask=b_mask.unsqueeze(1))
        b_output = b_output.squeeze(1)

        concat_output = torch.cat((word.squeeze(1), state.squeeze(1), segment.squeeze(1), c_output, b_output), dim=-1)

        feature_output=self.readout(concat_output)

        p = F.softmax(self.generator(feature_output), dim=-1)

        return p

class StateTracker(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super(StateTracker, self).__init__()

        self.linear=nn.Linear(hidden_size*2, hidden_size)
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=1, bidirectional=False, batch_first=True)

    def initialize(self, segment, state):
        return self.linear(torch.cat([state, segment], dim=-1))

    def forward(self, word, state):
        return self.gru(word, state.transpose(0, 1))[1].transpose(0,1)

class Mixturer(nn.Module):
    def __init__(self, hidden_size):
        super(Mixturer, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)

    def forward(self, state, dists1, dists2, dyn_map):
        p_k_v = torch.sigmoid(self.linear1(state.squeeze(1)))

        dists2 = torch.bmm(dists2.unsqueeze(1), dyn_map).squeeze(1)

        # dist = torch.cat([p_k_v * dists1, (1. - p_k_v) * dists2], dim=-1)

        return p_k_v * dists1+(1. - p_k_v) * dists2

# class Criterion(object):
#     def __init__(self, tgt_vocab_size, eps=1e-10):
#         super(Criterion, self).__init__()
#         self.eps = eps
#         self.offset = tgt_vocab_size
#
#     def __call__(self, gen_output, response, dyn_response, UNK, reduction='mean'):
#         dyn_not_pad = dyn_response.ne(0).float()
#         v_not_unk = response.ne(UNK).float()
#         v_not_pad=response.ne(0).float()
#
#         if len(gen_output.size()) > 2:
#             gen_output = gen_output.view(-1, gen_output.size(-1))
#
#         p_dyn = gen_output.gather(1, dyn_response.view(-1, 1) + self.offset).view(-1)
#         p_dyn = p_dyn.mul(dyn_not_pad.view(-1))
#
#         p_v = gen_output.gather(1, response.view(-1, 1)).view(-1)
#         p_v = p_v.mul(v_not_unk.view(-1))
#
#         p = p_dyn + p_v + self.eps
#         p = p.log()
#
#         loss = -p.mul(v_not_pad.view(-1))
#         if reduction=='mean':
#             return loss.sum()/v_not_pad.sum()
#         elif reduction=='none':
#             return loss.view(response.size())

class GLKS(EncDecModel):
    def __init__(self, min_window_size, num_windows, embedding_size, hidden_size, vocab2id, id2vocab, max_dec_len, beam_width, emb_matrix=None, eps=1e-10):
        super(GLKS, self).__init__(vocab2id=vocab2id, max_dec_len=max_dec_len, beam_width=beam_width, eps=eps)
        self.vocab_size = len(vocab2id)
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab

        self.b_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)
        self.c_encoder = GenEncoder(1, self.vocab_size, embedding_size, hidden_size, emb_matrix=emb_matrix)

        if emb_matrix is None:
            self.embedding = nn.Embedding(self.vocab_size, embedding_size, padding_idx=0)
        else:
            self.embedding=create_emb_layer(emb_matrix)

        self.state_tracker = StateTracker(embedding_size, hidden_size)

        self.k_selector = KnowledgeSelector(hidden_size, min_window_size=min_window_size, n_windows=num_windows)

        self.c_generator = CopyGenerator(embedding_size, hidden_size)
        self.v_generator = VocabGenerator(embedding_size, hidden_size, self.vocab_size)

        self.mixture = Mixturer(hidden_size)

        # self.criterion = Criterion(self.vocab_size)

    def encode(self, data):
        b_enc_outputs, b_states= self.b_encoder(data['background'])
        c_enc_outputs, c_states= self.c_encoder(data['context'])
        b_enc_output=b_enc_outputs[:,-1]
        b_state=b_states[:,-1].unsqueeze(1)
        c_enc_output=c_enc_outputs[:,-1]
        c_state = c_states[:, -1].unsqueeze(1)

        segment, p_s, p_g =self.k_selector(b_enc_output, c_enc_output, c_state, data['background'].ne(0), data['context'].ne(0))

        return {'b_enc_output': b_enc_output, 'b_state': b_state, 'c_enc_output': c_enc_output, 'c_state':c_state, 'segment':segment, 'p_s':p_s, 'p_g':p_g}

    def init_decoder_states(self, data, encode_outputs):
        return self.state_tracker.initialize(encode_outputs['segment'], encode_outputs['c_state'])

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs):
        word_embedding = F.dropout(self.embedding(previous_word), training=self.training).unsqueeze(1)

        states=previous_deocde_outputs['state']
        states=self.state_tracker(word_embedding, states)

        if 'p_k' in previous_deocde_outputs:
            p_k = previous_deocde_outputs['p_k']
            p_v = previous_deocde_outputs['p_v']
        else:
            p_k = None
            p_v = None

        p_k = self.c_generator(p_k, word_embedding, states, encode_outputs['segment'], encode_outputs['b_enc_output'], encode_outputs['c_enc_output'], data['background'].ne(0), data['context'].ne(0))
        p_v = self.v_generator(p_v, word_embedding, states, encode_outputs['segment'], encode_outputs['b_enc_output'], encode_outputs['c_enc_output'], data['background'].ne(0), data['context'].ne(0))

        return {'p_k':p_k, 'p_v':p_v, 'state':states}

    def generate(self, data, encode_outputs, decode_outputs, softmax=True):
        p = self.mixture(decode_outputs['state'], decode_outputs['p_v'], decode_outputs['p_k'], data['background_map'])
        return {'p': p}

    def generation_to_decoder_input(self, data, indices):
        return indices

    def to_word(self, data, gen_output, k=5, sampling=False):
        gen_output = gen_output['p']
        if not sampling:
            return topk(gen_output, k=k)
        else:
            return randomk(gen_output, k=k)

    def to_sentence(self, data, batch_indices):
        return to_sentence(batch_indices, self.id2vocab)

    def forward(self, data, method='mle_train'):
        data['background_map'] = build_map(data['background_map'], max=self.vocab_size)
        if 'train' in method:
            return self.do_train(data, type=method)
        elif method=='test':
            if self.beam_width==1:
                return {'answer': self.greedy(data)}
            else:
                return {'answer': self.beam(data)}

    def do_train(self, data, type='mle_train'):
        encode_output, init_decoder_state, all_decode_output, all_gen_output = decode_to_end(self, data, self.vocab2id, tgt=data['response'])
        loss=list()
        if 'mle' in type:
            p = torch.cat([p['p'].unsqueeze(1) for p in all_gen_output], dim=1)
            p = p.view(-1, p.size(-1))
            r_loss = F.nll_loss((p.reshape(-1, p.size(-1))+1e-8).log(), data['response'].reshape(-1), ignore_index=0).unsqueeze(0)
            loss+=[r_loss]
        if 'mce' in type:
            e1_loss = 1 - 0.1 * Categorical(probs=p + self.eps).entropy().mean().unsqueeze(0)
            loss += [e1_loss]
        if 'ds' in type:
            k_loss = F.kl_div((encode_output['p_s'].squeeze(1) + self.eps).log(), data['selection'] + self.eps, reduction='batchmean').unsqueeze(0)
            loss+=[k_loss]

        return loss
