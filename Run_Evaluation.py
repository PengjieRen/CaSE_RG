import os
import codecs
from evaluation.Eval_Rouge import *
from evaluation.Eval_Bleu import *
from evaluation.Eval_Meteor import *
from evaluation.Eval_Trec import *
from common.Utils import *
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--data_path", type=str)
args = parser.parse_args()

base_data_path = args.data_path
output_path='./output/'+args.model+'/result/'
cast_answer_file=base_data_path+'cast/cast.answer'
cast_qrel_file=base_data_path+'cast/cast.qrel'
marco_answer_file=base_data_path+'marco/marco.answer'
marco_qrel_file=base_data_path+'marco/marco.qrel'
quac_answer_file=base_data_path+'quac/quac.answer'
quac_qrel_file=base_data_path+'quac/quac.qrel'

tokenizer, vocab2id, id2vocab = bert_tokenizer()
detokenizer = bert_detokenizer()

def eval_all(gt_qrel_file, gt_answer_file, model_output_path):
    file_answer_dict=dict()
    file_run_dict=dict()
    files = os.listdir(model_output_path)
    for file in files:
        temp=file.split('.')
        if temp[-1]=='answer':
            if temp[0] in file_answer_dict:
                file_answer_dict[temp[0]].append(file)
            else:
                file_answer_dict[temp[0]]=[file]
        else:
            if temp[0] in file_run_dict:
                file_run_dict[temp[0]].append(file)
            else:
                file_run_dict[temp[0]]=[file]

    for key in file_run_dict:
        output_run = codecs.open(os.path.join(model_output_path, key + '.all.run'), mode='w', encoding='utf-8')
        for file in file_run_dict[key]:
            with codecs.open(os.path.join(model_output_path, file), encoding='utf-8') as f:
                for line in f:
                    output_run.write(line)
        output_run.close()

        output_file=os.path.join(model_output_path, key + '.all.run')
        # qrel_file=None
        # if 'cast' in key:
        #     qrel_file=cast_qrel_file
        # elif 'quac' in key:
        #     qrel_file = quac_qrel_file
        # elif 'marco' in key:
        #     qrel_file = marco_qrel_file

        rank_metrics = eval_trec_file(output_file, gt_qrel_file)
        print(key, rank_metrics)

    for key in file_answer_dict:
        output_answer = codecs.open(os.path.join(model_output_path, key + '.all.answer'), mode='w', encoding='utf-8')
        for file in file_answer_dict[key]:
            with codecs.open(os.path.join(model_output_path, file), encoding='utf-8') as f:
                for line in f:
                    output_answer.write(line)
        output_answer.close()

        output_file=os.path.join(model_output_path, key + '.all.answer')
        # answer_file = None
        # if 'cast' in key:
        #     answer_file = cast_answer_file
        # elif 'quac' in key:
        #     answer_file = quac_answer_file
        # elif 'marco' in key:
        #     answer_file = marco_answer_file

        rouges = eval_rouge_file(output_file, gt_answer_file, tokenizer, detokenizer)
        bleus = eval_bleu_file(output_file, gt_answer_file, tokenizer, detokenizer)
        meteors = eval_meteor_file(output_file, gt_answer_file, tokenizer, detokenizer)
        print(key, rouges, bleus, meteors)


eval_all(marco_qrel_file, marco_answer_file, output_path)
eval_all(quac_qrel_file, quac_answer_file, output_path)
eval_all(cast_qrel_file, cast_answer_file, output_path)