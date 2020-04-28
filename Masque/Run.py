import sys
sys.path.append('./')
from Masque.MasqueDataset import *
from torch import optim
from common.CumulativeTrainer import *
import torch.backends.cudnn as cudnn
import argparse
from Masque.Model import *
from Utils import *
from transformers.optimization import *

base_output_path = './output/Masque/'
dir_path = os.path.dirname(os.path.realpath(__file__))
embedding_size = 256
hidden_size=256
max_target_length=40
accumulation_steps=1
epoch=20

def train(args):
    batch_size = 16

    output_path = base_output_path
    dataset = args.dataset
    data_path = args.data_path + dataset + '/' + dataset

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    train_samples = torch.load(data_path + '.pkl')
    marco_train_size=len(train_samples)
    train_dataset = MasqueDataset(train_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(data_path + '.Masque.dataset.pkl'))

    model = Masque(max_target_length, id2vocab, vocab2id, hidden_size)
    init_params(model)

    model_bp_count = (epoch * marco_train_size) / (4 * batch_size * accumulation_steps)
    model_optimizer = optim.Adam(model.parameters(), lr=2.5e-4)
    model_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(model_optimizer, 2000, int(model_bp_count) + 100)
    model_trainer = CumulativeTrainer(model, tokenizer, detokenizer, args.local_rank, 4,
                                      accumulation_steps=accumulation_steps)

    for i in range(epoch):
        model_trainer.train_epoch('train', train_dataset, collate_fn, batch_size, i, model_optimizer, model_scheduler)
        model_trainer.serialize(i, output_path=output_path)

def test(args):
    batch_size = 16

    output_path = base_output_path
    dataset = args.dataset
    data_path = args.data_path + dataset + '/' + dataset

    tokenizer, vocab2id, id2vocab = bert_tokenizer()
    detokenizer = bert_detokenizer()

    marco_dev_samples = torch.load(data_path + 'marco/marco.dev.pkl')
    marco_dev_dataset = MasqueDataset(marco_dev_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(data_path + 'marco/marco.dev.Masque.dataset.pkl'))

    marco_test_samples = torch.load(data_path + 'marco/marco.test.pkl')
    marco_test_dataset = MasqueDataset(marco_test_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(data_path + 'marco/marco.test.Masque.dataset.pkl'))

    cast_test_samples = torch.load(data_path + '.pkl')
    cast_test_dataset = MasqueDataset(cast_test_samples, None, None, None, None, None, None, None, None, None, None, sample_tensor=torch.load(data_path + '.Masque.dataset.pkl'))

    for i in range(epoch):
        print('epoch', i)
        file = output_path + 'model/' + str(i) + '.pkl'

        if os.path.exists(file):
            model = Masque(max_target_length, id2vocab, vocab2id, hidden_size)
            model.load_state_dict(torch.load(file, map_location='cpu'))
            trainer = CumulativeTrainer(model, tokenizer, detokenizer, args.local_rank, 4)
            predictions=trainer.predict('test', marco_dev_dataset, collate_fn, batch_size)
            save_result(predictions, marco_dev_dataset, model.to_sentence, detokenizer, output_path, args.local_rank, i, 'marco_dev')
            predictions =trainer.predict('test', marco_test_dataset, collate_fn, batch_size)
            save_result(predictions, marco_test_dataset, model.to_sentence, detokenizer, output_path, args.local_rank, i, 'marco_test')
            predictions = trainer.predict('test', cast_test_dataset, collate_fn, batch_size)
            save_result(predictions, cast_test_dataset, model.to_sentence, detokenizer, output_path, args.local_rank, i, 'cast_test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--mode", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str)
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