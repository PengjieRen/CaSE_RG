import sys
import os
sys.path.append('./')
from CaSE.CaSEDataset import *
from torch import optim
from common.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from CaSE.Model import *
from Utils import *
from transformers.optimization import *

def train(args):
    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    data_path=os.path.join(args.data_path, args.dataset+ '/')

    train_samples = torch.load(os.path.join(data_path, args.dataset+ '.train.pkl'))
    train_size=len(train_samples)
    train_dataset = CaSEDataset(train_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(os.path.join(data_path,  args.dataset+ '.CaSE.dataset.pkl')))

    model = CaSE(args.max_span_size, args.max_target_length, id2vocab, vocab2id, args.hidden_size)
    init_params(model)

    model_bp_count = (args.epoch * train_size) / (args.num_gpu * args.batch_size * args.accumulation_steps)
    model_optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
    model_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(model_optimizer, 2000, int(model_bp_count) + 100)
    model_trainer = CumulativeTrainer(model, tokenizer, detokenizer, args.local_rank, args.num_gpu, accumulation_steps=args.accumulation_steps)

    for i in range(args.epoch):
        model_trainer.train_epoch('train', train_dataset, collate_fn, args.batch_size, i, model_optimizer, model_scheduler)
        model_trainer.serialize(i, output_path=args.output_path)

def test(args):
    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    data_path=os.path.join(args.data_path, args.dataset+ '/')

    dev_samples = torch.load(os.path.join(data_path, args.dataset+ '.dev.pkl'))
    if len(dev_samples)>10:
        dev_dataset = CaSEDataset(dev_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(os.path.join(data_path, args.dataset+ '.dev.CaSE.dataset.pkl')))

    test_samples = torch.load(os.path.join(data_path, args.dataset+ '.test.pkl'))
    if len(test_samples) > 10:
        test_dataset = CaSEDataset(test_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(os.path.join(data_path, args.dataset+ '.test.CaSE.dataset.pkl')))

    for i in range(args.epoch):
        print('epoch', i)
        file = args.output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = CaSE(args.max_span_size, args.max_target_length, id2vocab, vocab2id, args.hidden_size)
            model.load_state_dict(torch.load(file, map_location='cpu'))
            trainer = CumulativeTrainer(model, tokenizer, detokenizer, args.local_rank, args.num_gpu)
            if dev_dataset:
                predictions=trainer.predict('test', dev_dataset, collate_fn, args.batch_size)
                save_result(predictions, dev_dataset, model.to_sentence, detokenizer, args.output_path, args.local_rank, i, args.dataset+'_dev')
            if test_dataset:
                predictions =trainer.predict('test', test_dataset, collate_fn, args.batch_size)
                save_result(predictions, test_dataset, model.to_sentence, detokenizer, args.output_path, args.local_rank, i, args.dataset+'_test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_path", type=str, default='./dataset/')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--output_path", type=str, default='./output/CaSE/')
    parser.add_argument("--embedding_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--max_span_size", type=int, default=4)
    parser.add_argument("--max_target_length", type=int, default=40)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_gpu", type=int, default=4)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.distributed.init_process_group(backend='NCCL', init_method='env://')

    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    print(torch.__version__)
    print(torch.version.cuda)
    print(cudnn.version())

    init_seed(123456)

    if args.mode=='test':
        test(args)
    elif args.mode=='train':
        train(args)