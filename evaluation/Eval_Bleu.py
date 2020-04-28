import codecs
from nltk.translate.bleu_score import *

def rounder(num):
    return round(num, 2)

def eval_bleu_file(run_file, ref_file, tokenizer, detokenizer):
    run_dict={}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)==4:
                run_dict[temp[1]+'##<>##'+temp[2]]= tokenizer(temp[3])
    ref_dict = {}
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t')
            if len(temp) == 4:
                tokenized = tokenizer(temp[3])
                if temp[1] in ref_dict:
                    ref_dict[temp[1]].append(tokenized)
                else:
                    ref_dict[temp[1]]=[tokenized]

    bleu=0.
    for id in run_dict:
        bleu+=sentence_bleu(ref_dict[id.split('##<>##')[0]],run_dict[id])
    return {'BLEU': rounder(bleu*100/len(run_dict))}

