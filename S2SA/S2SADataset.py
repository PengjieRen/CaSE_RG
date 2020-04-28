from torch.utils.data import Dataset
from common.Utils import *
from torch.nn.utils.rnn import pad_sequence

class S2SADataset(Dataset):
    def __init__(self, samples, query, passage, vocab2id, num_passage=10, context_len=80, passage_len=200, answer_len=50, n=1E10, sample_tensor=None):
        super(S2SADataset, self).__init__()

        if sample_tensor is None:
            self.passage_len=passage_len
            self.num_passage=num_passage
            self.context_len=context_len
            self.answer_len=answer_len

            self.samples=samples
            self.query=query
            self.passage=passage

            self.vocab2id=vocab2id
            self.n=n

            self.sample_tensor=[]
            self.load()
        else:
            self.samples=samples
            self.sample_tensor=sample_tensor
            self.len=len(self.sample_tensor)

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
            background = []
            for p in passage:
                background += p
            background_tensor=torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in background], requires_grad=False).long()

            response= (sample['answer']+[EOS_WORD])[:self.answer_len]
            response_tensor =torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response], requires_grad=False).long()

            self.sample_tensor.append([id_tensor, query_tensor, background_tensor, response_tensor])
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
    id,context,background,response = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True),
            'background': pad_sequence(background, batch_first=True)}

