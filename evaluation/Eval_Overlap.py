import codecs
import os

def rounder(num):
    return round(num, 2)

def ngram(words, n):
    ngrams=set()
    for i in range(len(words)):
        if i+n <= len(words):
            ngrams.add(' '.join(words[i:i+n]))
    return ngrams

def overlap_ratio(answer, passage):
    if len(answer)==0:
        return 0
    return float(len(answer.intersection(passage)))/float(len(answer))

def eval_overlap_file(run_file, passages, samples, tokenizer, detokenizer):
    run_dict=dict()
    with codecs.open(run_file, encoding='utf-8') as f:
        for line in f:
            temp=line.strip('\n').strip('\r').split('\t')
            run_dict[temp[1]+'##<>##'+temp[2]]=tokenizer(temp[3])
    passage_dict=dict()
    for sample in samples:
        key=sample['query_id']+'##<>##'+';'.join(sample['passage_id'])
        if key not in run_dict:
            continue
        passage_token=[]
        for p_id in sample['passage_pool_id']:
            passage_token+=passages[p_id]
        passage_dict[key]=passage_token

    overlap1 = 0
    overlap2 = 0
    overlap3 = 0
    overlap4 = 0
    for key in run_dict:
        answer1=set(run_dict[key])
        passage1=set(passage_dict[key])
        overlap1+=overlap_ratio(answer1, passage1)

        answer2 = ngram(run_dict[key], 2)
        passage2 = ngram(passage_dict[key], 2)
        overlap2 += overlap_ratio(answer2, passage2)

        answer3 = ngram(run_dict[key], 3)
        passage3 = ngram(passage_dict[key], 3)
        overlap3 += overlap_ratio(answer3, passage3)

        answer4 = ngram(run_dict[key], 4)
        passage4 = ngram(passage_dict[key], 4)
        overlap4 += overlap_ratio(answer4, passage4)

    overlap1=rounder(overlap1*100/len(run_dict))
    overlap2 = rounder(overlap2 * 100 / len(run_dict))
    overlap3 = rounder(overlap3 * 100 / len(run_dict))
    overlap4 = rounder(overlap4 * 100 / len(run_dict))

    return {'Overlap-1': overlap1, 'Overlap-2': overlap2, 'Overlap-3':overlap3, 'Overlap-4':overlap4}


