from torch.utils.data import Dataset
from common.Utils import *
from torch.nn.utils.rnn import pad_sequence

class TMemNetDataset(Dataset):
    def __init__(self, samples, query, passage, vocab2id, num_passage=10, context_len=80, passage_len=200, answer_len=50, n=1E10, sample_tensor=None):
        super(TMemNetDataset, self).__init__()

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

            contexts = [self.query[cid] + [SEP_WORD] for cid in sample['context_id']]
            while len(contexts) < 3:
                contexts = [[UNK_WORD] + [SEP_WORD]] + contexts
            contexts = contexts[-3:]
            context = []
            for q in contexts:
                context += q
            query = [CLS_WORD] + context + self.query[sample['query_id']]
            query = query[-self.context_len:]
            query_tensor = torch.tensor(
                [self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in query],
                requires_grad=False).long()

            passage=[self.passage[pid][:self.passage_len] + [PAD_WORD] * (self.passage_len - len(self.passage[pid])) if pid in self.passage and len(self.passage[pid])>0 else [UNK_WORD] + [PAD_WORD] * (self.passage_len - 1) for pid in sample['passage_pool_id']]
            while len(passage)!=self.num_passage:
                passage.append([UNK_WORD] + [PAD_WORD] * (self.passage_len - 1))
            passage = passage[:self.num_passage]
            passage_tensor=[torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in p], requires_grad=False).long() for p in passage]
            passage_tensor = torch.stack(passage_tensor)

            passage_label = [torch.tensor([sample['passage_pool_id'].index(pid)], requires_grad=False).long() for pid in sample['passage_id']]

            response= (sample['answer']+[EOS_WORD])[:self.answer_len]
            response_tensor =torch.tensor([self.vocab2id.get(w) if w in self.vocab2id else self.vocab2id[UNK_WORD] for w in response], requires_grad=False).long()

            self.sample_tensor.append([id_tensor, query_tensor, passage_tensor, passage_label, response_tensor])
            self.len = id + 1
            if id>=self.n:
                break

    def __getitem__(self, index):
        sample= self.sample_tensor[index]
        return [sample[0], sample[1], sample[2], sample[3][random.randint(0, len(sample[3]) - 1)], sample[4]]

    def __len__(self):
        return self.len

    def context_id(self,id):
        return self.samples[id]['context_id']

    def query_id(self,id):
        return self.samples[id]['query_id']

    def passage_id(self,id):
        return self.samples[id]['passage_id']

    def pool(self, id):
        return self.samples[id]['passage_pool_id']
def collate_fn(data):
    id,context,passage,passage_label,response = zip(*data)

    return {'id': torch.cat(id),
            'context': pad_sequence(context, batch_first=True),
            'response': pad_sequence(response, batch_first=True),
            'label': torch.cat(passage_label),
            'passage': torch.stack(passage)}

