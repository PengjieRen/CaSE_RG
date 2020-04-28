# coding: utf-8
import torch.nn.functional as F
from common.Utils import *
from torch.distributions.categorical import *
from common.Constants import *

def sample(model, data, vocab2id, max_len=20, encode_outputs=None, init_decoder_states=None):
    BOS=vocab2id[BOS_WORD]
    EOS=vocab2id[EOS_WORD]
    UNK=vocab2id[UNK_WORD]
    PAD=vocab2id[PAD_WORD]

    batch_size = data['id'].size(0)

    if encode_outputs is None:
        encode_outputs = model.encode(data)

    if init_decoder_states is None:
        init_decoder_states = model.init_decoder_states(data, encode_outputs)

    init_decoder_input = new_tensor([BOS] * batch_size, requires_grad=False)

    indices = list()
    end = new_tensor([0] * batch_size).long() == 1

    decoder_input = init_decoder_input
    all_gen_outputs = list()
    all_decode_outputs = [dict({'state': init_decoder_states})]

    # ranp=random.randint(0, max_len-1)
    for t in range(max_len):
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1]
        )

        gen_output = model.generate(data, encode_outputs, decode_outputs, softmax=True)

        all_gen_outputs.append(gen_output.unsqueeze(0))
        all_decode_outputs.append(decode_outputs)

        probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=True)
        # if random.uniform(0,1)>0.9:
        #     probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=True)
        # else:
        #     probs, ids = model.to_word(data, F.softmax(gen_output, dim=1), 1, sampling=False)

        indice = ids[:, 0]
        this_end = indice == EOS
        if t == 0:
            indice.masked_fill_(this_end, UNK)
        elif t==max_len-1:
            indice[:]=EOS
            indice.masked_fill_(end, PAD)
        else:
            indice.masked_fill_(end, PAD)
        indices.append(indice.unsqueeze(1))
        end = end | this_end

        decoder_input = model.generation_to_decoder_input(data, indice)

    all_gen_outputs = torch.cat(all_gen_outputs, dim=0).transpose(0, 1).contiguous()

    return torch.cat(indices, dim=1), encode_outputs, init_decoder_states, all_decode_outputs, all_gen_outputs


def greedy(model,data,vocab2id,max_len=20, encode_outputs=None, init_decoder_states=None):
    BOS = vocab2id[BOS_WORD]
    EOS = vocab2id[EOS_WORD]
    UNK = vocab2id[UNK_WORD]
    PAD = vocab2id[PAD_WORD]

    batch_size=data['id'].size(0)

    if encode_outputs is None:
        encode_outputs= model.encode(data)

    if init_decoder_states is None:
        decoder_states = model.init_decoder_states(data, encode_outputs)
    else:
        decoder_states=init_decoder_states

    decoder_input = new_tensor([BOS] * batch_size, requires_grad=False)
    all_decode_outputs = [dict({'state': decoder_states})]

    greedy_indices=list()
    greedy_end = new_tensor([0] * batch_size).long() == 1
    for t in range(max_len):
        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, all_decode_outputs[-1]
        )

        gen_output=model.generate(data, encode_outputs, decode_outputs, softmax=True)

        probs, ids=model.to_word(data, gen_output, 1)

        all_decode_outputs.append(decode_outputs)

        greedy_indice = ids[:,0]
        greedy_this_end = greedy_indice == EOS
        if t == 0:
            greedy_indice.masked_fill_(greedy_this_end, UNK)
        else:
            greedy_indice.masked_fill_(greedy_end, PAD)
        greedy_indices.append(greedy_indice.unsqueeze(1))
        greedy_end = greedy_end | greedy_this_end

        decoder_input = model.generation_to_decoder_input(data, greedy_indice)

    greedy_indice=torch.cat(greedy_indices,dim=1)
    return greedy_indice

def beam(model,data,vocab2id,max_len=20,width=5, encode_outputs=None, init_decoder_states=None):
    BOS = vocab2id[BOS_WORD]
    EOS = vocab2id[EOS_WORD]
    UNK = vocab2id[UNK_WORD]
    PAD = vocab2id[PAD_WORD]

    batch_size = data['id'].size(0)

    if encode_outputs is None:
        encode_outputs = model.encode(data)

    if init_decoder_states is None:
        decoder_states = model.init_decoder_states(data, encode_outputs)
    else:
        decoder_states=init_decoder_states

    decode_outputs = dict({'state':decoder_states})

    next_fringe = []
    results = dict()
    for i in range(batch_size):
        next_fringe += [Node(parent=None, state=get_data(i, decode_outputs), word=BOS_WORD, value=BOS, cost=0.0, encode_outputs=get_data(i, encode_outputs), data=get_data(i,data), batch_id=i)]
        results[i] = []

    for l in range(max_len+1):
        fringe = []
        for n in next_fringe:
            if n.value == EOS or l == max_len:
                results[n.batch_id].append(n)
            else:
                fringe.append(n)

        if len(fringe) == 0:
            break

        data=concat_data([n.data for n in fringe])
        decode_outputs=concat_data([n.state for n in fringe])
        encode_outputs = concat_data([n.encode_outputs for n in fringe])

        decoder_input= new_tensor([n.value for n in fringe], requires_grad=False)
        decoder_input = model.generation_to_decoder_input(data, decoder_input)

        decode_outputs = model.decode(
            data, decoder_input, encode_outputs, decode_outputs
        )

        gen_output = model.generate(data, encode_outputs, decode_outputs, softmax=True)

        probs, ids = model.to_word(data, gen_output, width)

        next_fringe_dict = dict()
        for i in range(batch_size):
            next_fringe_dict[i] = []

        for i in range(len(fringe)):
            n = fringe[i]

            for j in range(width):
                loss = -math.log(probs[i,j].item() + 1e-10)

                n_new = Node(parent=n, state=get_data(i, decode_outputs), word=None, value=ids[i,j].item(), cost=loss,
                             encode_outputs=n.encode_outputs,
                             data=n.data, batch_id=n.batch_id)

                next_fringe_dict[n_new.batch_id].append(n_new)

        next_fringe = []
        for i in range(batch_size):
            next_fringe += sorted(next_fringe_dict[i], key=lambda n: n.cum_cost / n.length)[:width]

    outputs = []
    for i in range(batch_size):
        results[i].sort(key=lambda n: n.cum_cost / n.length)
        outputs.append(results[i][0])# currently only select the first one

    # sents=[node.to_sequence_of_words()[1:-1] for node in outputs]
    indices=merge1D([new_tensor(node.to_sequence_of_values()[1:]) for node in outputs])

    return indices

class Node(object):
    def __init__(self, parent, state, word, value, cost, encode_outputs, data, batch_id=None):
        super(Node, self).__init__()
        self.word=word
        self.value = value
        self.parent = parent # parent Node, None for root
        self.state = state
        self.cum_cost = parent.cum_cost + cost if parent else cost # e.g. -log(p) of sequence up to current node (including)
        self.length = 1 if parent is None else parent.length + 1
        self.encode_outputs = encode_outputs # can hold, for example, attention weights
        self._sequence = None
        self.batch_id=batch_id
        self.data=data

    def to_sequence(self):
        # Return sequence of nodes from root to current node.
        if not self._sequence:
            self._sequence = []
            current_node = self
            while current_node:
                self._sequence.insert(0, current_node)
                current_node = current_node.parent
        return self._sequence

    def to_sequence_of_values(self):
        return [s.value for s in self.to_sequence()]

    def to_sequence_of_words(self):
        return [s.word for s in self.to_sequence()]
