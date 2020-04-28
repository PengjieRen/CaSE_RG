import torch
import os
import time
import numpy as np
from common.Utils import *
from CaSE.CaSEDataset import *
from GLKS.GLKSDataset import *
from GTTP.GTTPDataset import *
from S2SA.S2SADataset import *
from TMemNet.TMemNetDataset import *
from Masque.MasqueDataset import *

query_len=60
passage_len = 100
max_span_size=4
num_passage=10
max_target_length=40
min_window_size = 4
num_windows = 1
base_data_path='../dataset/'

init_seed(123456)

tokenizer, vocab2id, id2vocab = bert_tokenizer()
detokenizer = bert_detokenizer()

print('Item size', len(vocab2id))

vocab2id_, id2vocab_, id2freq_=load_vocab(os.path.join(base_data_path + 'marco/', 'marco.vocab'))
id2freq=dict()
for id_ in id2freq_:
    w=id2vocab_[id_]
    if w in vocab2id:
        id=vocab2id[w]
        id2freq[id]=id2freq_[id_]

def get_ms():
    return time.time() * 1000

def init_seed(seed=None):
    if seed is None:
        seed = int(get_ms() // 1000)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_answer(file, tokenizer):
    answer=[]
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)>=4:
                if len(temp[0])<1:
                    temp[0]=[]
                else:
                    temp[0] = temp[0].split(';')
                temp[2] = temp[2].split(';')
                temp[3] = tokenizer(temp[3])
                answer.append(temp)
    return answer

def load_passage(file, pool, tokenizer):
    poolset=set()
    if pool is not None:
        for k in pool:
            poolset.update(pool[k])
    passage=dict()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)==2 and temp[0] in poolset:
                passage[temp[0]]=' [SEP] '.join([' '.join(tokenizer(sent)) for sent in nltk.sent_tokenize(temp[1])]).split(' ')
    return passage

def load_pool(file, topk=10):
    pool={}
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp=line.strip('\n').strip('\r').split(' ')
            if len(temp) == 6:
                if temp[0] not in pool:
                    pool[temp[0]]=[temp[2]]
                else:
                    if len(pool[temp[0]])==topk:
                        continue
                    pool[temp[0]].append(temp[2])
    return pool

def load_qrel(file):
    qrel = dict()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp = line.strip('\n').strip('\r').split(' ')
            if len(temp) == 4:
                if int(temp[3])>0:
                    qrel[temp[0]]=temp[2]
    return qrel

def load_query(file, tokenizer):
    query=dict()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)==2:
                query[temp[0]]=tokenizer(temp[1])
    return query

def load_split(file):
    train=set()
    dev=set()
    test=set()
    with codecs.open(file, encoding='utf-8') as f:
        next(f)
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)==2:
                if temp[1]=='train':
                    train.add(temp[0])
                elif temp[1]=='dev':
                    dev.add(temp[0])
                elif temp[1]=='test':
                    test.add(temp[0])
    return train, dev, test

def split_data(split_file, samples):
    train, dev, test=load_split(split_file)
    train_samples=list()
    dev_samples=list()
    test_samples=list()
    for sample in samples:
        if sample['query_id'] in train:
            train_samples.append(sample)
        elif sample['query_id'] in dev:
            dev_samples.append(sample)
        elif sample['query_id'] in test:
            test_samples.append(sample)
    return train_samples, dev_samples, test_samples

def load_default(answer_file, passage_file, pool_file, qrel_file, query_file, query_reformation_file, tokenizer, topk=10, randoms=1):
    random.seed(1)
    answer=load_answer(answer_file, tokenizer)
    pool=None
    if pool_file is not None:
        pool=load_pool(pool_file, 10*topk)
    query=load_query(query_file, tokenizer)
    qrel=load_qrel(qrel_file)
    reformulated_query=None
    if query_reformation_file and os.path.exists(query_reformation_file):
        reformulated_query=load_query(query_reformation_file, tokenizer)

    samples=[]
    for i in range(len(answer)):
        for j in range(randoms):
            c_id, q_id, p_id, ans = answer[i][:4]
            q_pool=None
            if pool is not None:
                q_pool=pool[q_id]
                random.shuffle(q_pool)

            sample = dict()
            sample['context_id'] = c_id
            sample['query_id'] = q_id
            sample['passage_id'] = p_id.copy()
            sample['answer'] = ans
            sample['passage_pool_id'] = p_id.copy()
            if q_pool is not None:
                for p in p_id:
                    if p not in q_pool:
                        q_pool.append(p)
            q_qrel = dict()
            if q_id in qrel:
                q_qrel = qrel[q_id]
            if q_pool is not None:
                for p in q_pool:
                    if len(sample['passage_pool_id']) == topk:
                        break
                    if p not in sample['passage_pool_id'] and p not in q_qrel:
                        sample['passage_pool_id'].append(p)
            random.shuffle(sample['passage_pool_id'])
            sample['qrel_file'] = qrel_file
            sample['answer_file'] = answer_file
            sample['passage_file'] = passage_file
            sample['pool_file'] = pool_file
            sample['query_file'] = query_file
            sample['query_reformation_file'] = query_reformation_file
            samples.append(sample)

    passage = load_passage(passage_file, pool, tokenizer)
    print(len(samples), 'samples')
    return samples, query, reformulated_query, passage

def merge_test(samples):
    rs=dict()
    for sample in samples:
        id='-'.join(sample['context_id'])+'_'+sample['query_id']+'_'+'-'.join(sample['passage_pool_id'])
        if id not in rs:
            rs[id]=sample.copy()
    return list(rs.values())

if __name__ == '__main__':
    dataset = 'cast'
    if os.path.exists(base_data_path + dataset+ '/train.pkl'):
        query = torch.load(base_data_path + dataset+'/' +dataset+ '.query.pkl')
        passage = torch.load(base_data_path + dataset+'/' +dataset+ '.passage.pkl')
        reformulated_query = torch.load(base_data_path + dataset+'/' +dataset+ '.reformulation.query.pkl')

        train_samples = torch.load(base_data_path + dataset+'/' +dataset+ '.train.pkl')
        dev_samples = torch.load(base_data_path + dataset+'/' +dataset+ '.dev.pkl')
        test_samples = torch.load(base_data_path + dataset+'/' +dataset+ '.test.pkl')
    else:
        samples, query, reformulated_query, passage = load_default(base_data_path + dataset+'/' +dataset+ '.answer',
                                                                                  base_data_path + dataset+'/' +dataset+ '.passage',
                                                                                  base_data_path + dataset+'/' +dataset+ '.pool',
                                                                                  base_data_path + dataset+'/' +dataset+ '.qrel',
                                                                                  base_data_path + dataset+'/' +dataset+ '.query',
                                                                                  base_data_path + dataset+'/' +dataset+ '.reformulation.query',
                                                                                  tokenizer)
        train_samples, dev_samples, test_samples = split_data(base_data_path + dataset+'/' +dataset+ '.split', samples)

        dev_samples=merge_test(dev_samples)
        test_samples=merge_test(test_samples)

        torch.save(query, base_data_path + dataset+'/' +dataset+ '.query.pkl')
        torch.save(passage, base_data_path + dataset+'/' +dataset+ '.passage.pkl')
        torch.save(reformulated_query, base_data_path + dataset+'/' +dataset+ '.reformulated_query.pkl')
        torch.save(train_samples, base_data_path + dataset+'/' +dataset+ '.train.pkl')
        torch.save(dev_samples, base_data_path + dataset+'/' +dataset+ '.dev.pkl')
        torch.save(test_samples, base_data_path + dataset+'/' +dataset+ '.test.pkl')

    print('Data size',  len(train_samples), len(dev_samples), len(test_samples))

    if len(train_samples)>0:
        train_dataset= CaSEDataset(train_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.train.CaSE.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset= CaSEDataset(dev_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.dev.CaSE.dataset.pkl')

    if len(test_samples)>0:
        test_dataset= CaSEDataset(test_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.test.CaSE.dataset.pkl')

    if len(train_samples)>0:
        train_dataset= MasqueDataset(train_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.train.Masque.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset= MasqueDataset(dev_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.dev.Masque.dataset.pkl')

    if len(test_samples)>0:
        test_dataset= MasqueDataset(test_samples, query, passage, vocab2id, id2vocab, id2freq, num_passage, query_len, passage_len, max_span_size, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.test.Masque.dataset.pkl')

    if len(train_samples)>0:
        train_dataset = GLKSDataset(train_samples, query, passage, vocab2id, min_window_size, num_windows, num_passage, query_len, passage_len, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.train.GLKS.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset = GLKSDataset(dev_samples, query, passage, vocab2id, min_window_size, num_windows, num_passage, query_len, passage_len, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.dev.GLKS.dataset.pkl')

    if len(test_samples)>0:
        test_dataset = GLKSDataset(test_samples, query, passage, vocab2id, min_window_size, num_windows, num_passage, query_len, passage_len, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.test.GLKS.dataset.pkl')

    if len(train_samples)>0:
        train_dataset = GTTPDataset(train_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.train.GTTP.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset = GTTPDataset(dev_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.dev.GTTP.dataset.pkl')

    if len(test_samples)>0:
        test_dataset = GTTPDataset(test_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset+'/' +dataset+ '.test.GTTP.dataset.pkl')

    if len(train_samples)>0:
        train_dataset = S2SADataset(train_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.train.S2SA.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset = S2SADataset(dev_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.dev.S2SA.dataset.pkl')

    if len(test_samples)>0:
        test_dataset = S2SADataset(test_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.test.S2SA.dataset.pkl')

    if len(train_samples)>0:
        train_dataset = TMemNetDataset(train_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(train_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.train.TMemNet.dataset.pkl')

    if len(dev_samples)>0:
        dev_dataset = TMemNetDataset(dev_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(dev_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.dev.TMemNet.dataset.pkl')

    if len(test_samples)>0:
        test_dataset = TMemNetDataset(test_samples, query, passage, vocab2id, num_passage, query_len, passage_len, max_target_length)
        torch.save(test_dataset.sample_tensor, base_data_path + dataset + '/' + dataset + '.test.TMemNet.dataset.pkl')