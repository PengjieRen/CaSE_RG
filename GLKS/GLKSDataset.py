from torch.utils.data import Dataset
from common.Utils import *
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def get_selection_label(b,r, min_window_size=5, n_windows=4):
    window_size = min_window_size
    bs = list()
    for i in range(n_windows):
        bs.append(F.pad(b.unfold(1, window_size, min_window_size), (0, min_window_size*n_windows - window_size)))
        window_size += min_window_size
    b_segments= torch.cat(bs, dim=1)

    b_list=b_segments.tolist()
    r_list=r.tolist()

    overlap=[[len(set(seg).intersection(r_list[i])) for seg in b_list[i]] for i in range(len(b_list))]

    p_s=F.softmax(torch.tensor(overlap).float(), dim=-1).detach()
    return p_s

class GLKSDataset(Dataset):
    def __init__(self, samples, query, passage, vocab2id, min_window_size=5, num_windows=4, num_passage=10, context_len=60, passage_len=200, answer_len=40, n=1E10, sample_tensor=None):
        super(GLKSDataset, self).__init__()

        if sample_tensor is None:
            self.min_window_size=min_window_size
            self.num_windows=num_windows
            self.passage_len=passage_len
            self.context_len=context_len
            self.num_passage=num_passage
            self.answer_len=answer_len

            self.samples=samples
            self.query=query
            self.passage=passage

            self.vocab2id=vocab2id
            self.n=n

            self.sample_tensor=[]
            self.load()
        else:
            self.samples = samples
            self.sample_tensor = sample_tensor
            self.len = len(self.sample_tensor)

    def load(self):
        for id in range(len(self.samples)):
            sample=self.samples[id]
            id_tensor=torch.tensor([id]).long()

            contexts = [self.query[cid] for cid in sample['context_id']]
            context = []
            for q in contexts:
                context += q
            query = [CLS_WORD] + context + [SEP_WORD] + self.query[sample['query_id']]
            if len(query) > self.context_len:
                query = query[-self.context_len:]
            elif len(query) < self.context_len:
                query = query + [PAD_WORD] * (self.context_len - len(query))
            query_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in query],
                requires_grad=False).long()

            passage = []
            for pid in sample['passage_pool_id']:
                if pid in self.passage:
                    p = [CLS_WORD] + self.passage[pid] + [SEP_WORD]
                    if len(p) > self.passage_len:
                        p = p[:self.passage_len - 1] + [SEP_WORD]
                    elif len(p) < self.passage_len:
                        p = p + [SEP_WORD] + [PAD_WORD] * (self.passage_len - len(p) - 1)
                    passage.append(p)
            while len(passage) < self.num_passage:
                passage.append([CLS_WORD] + [SEP_WORD] + [PAD_WORD] * (self.passage_len - 2))
            background=[]
            for p in passage:
                background+=p
            background_tensor=torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in background], requires_grad=False).long()

            response= (sample['answer']+[EOS_WORD])[:self.answer_len]
            response_tensor =torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response], requires_grad=False).long()

            selection_tensor = get_selection_label(background_tensor.unsqueeze(0), response_tensor.unsqueeze(0), min_window_size=self.min_window_size, n_windows=self.num_windows)

            source_map_tensor = torch.tensor([self.vocab2id.get(w, self.vocab2id.get(UNK_WORD)) for w in background], requires_grad=False).long()

            self.sample_tensor.append([id_tensor, query_tensor, background_tensor, response_tensor, source_map_tensor, selection_tensor])
            self.len = id + 1
            if id>=self.n:
                break
        print('data size: ', self.len)

    def __getitem__(self, index):
        return self.sample_tensor[index]

    def __len__(self):
        return self.len

    def context_id(self,id):
        return self.samples[id]['context_id']

    def query_id(self,id):
        return self.samples[id]['query_id']

    def passage_id(self,id):
        return self.samples[id]['passage_id']

def collate_fn(data):
    id,context,background,response,source_map, selection = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True),
            'background': pad_sequence(background, batch_first=True),
            'background_map': pad_sequence(source_map, batch_first=True),
            'selection': torch.cat(selection)}

