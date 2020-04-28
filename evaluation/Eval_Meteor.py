import codecs
from nltk.translate.meteor_score import *

def rounder(num):
    return round(num, 2)

def eval_meteor_file(run_file, ref_file, tokenizer=None, detokenizer=None):
    run_dict={}
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            if len(temp)==4:
                run_dict[temp[1]+'##<>##'+temp[2]]= detokenizer(tokenizer(temp[3]))
    ref_dict = {}
    with codecs.open(ref_file, encoding='utf-8') as f:
        for line in f:
            temp = line.strip('\n').strip('\r').split('\t')
            if len(temp) == 4:
                tokenized = detokenizer(tokenizer(temp[3]))
                if temp[1] in ref_dict:
                    ref_dict[temp[1]].append(tokenized)
                else:
                    ref_dict[temp[1]]=[tokenized]

        meteor=0.
    for id in run_dict:
        meteor+=meteor_score(ref_dict[id.split('##<>##')[0]],run_dict[id])
    return {'METEOR': rounder(meteor*100/len(run_dict))}

