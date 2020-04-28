from common.Generations import *

class EncDecModel(nn.Module):
    def __init__(self, vocab2id, max_dec_len=120, beam_width=1, eps=1e-10):
        super(EncDecModel, self).__init__()
        self.eps=eps
        self.beam_width=beam_width
        self.max_dec_len=max_dec_len
        self.vocab2id=vocab2id

    def encode(self, data):
        raise NotImplementedError

    def init_decoder_states(self, data, encode_output):
        return None

    def decode(self, data, previous_word, encode_outputs, previous_deocde_outputs, decode_step):
        raise NotImplementedError

    def generate(self, data, encode_outputs, decode_outputs, softmax=False):
        raise NotImplementedError

    def to_word(self, data, gen_outputs, k=5, sampling=False):
        raise NotImplementedError

    def generation_to_decoder_input(self, data, indices):
        return indices

    def loss(self,data, encode_output,decode_outputs, gen_outputs, reduction='mean'):
        raise NotImplementedError

    def to_sentence(self, data, batch_indice):
        raise NotImplementedError

    def sample(self, data):
        raise NotImplementedError

    def greedy(self, data):
        return greedy(self,data,self.vocab2id, self.max_dec_len)

    def beam(self, data):
        return beam(self, data, self.vocab2id, self.max_dec_len, self.beam_width)