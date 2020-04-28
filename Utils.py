import codecs
import os
from common.Utils import *

def save_result(predictions, dataset, to_sentence, detokenizer, output_path, local_rank, epoch, eval_type):
    def sort(run):
        run = sorted(run, key=lambda r: r[4], reverse=True)
        for i in range(len(run)):
            run[i][3] = str(i + 1)
            run[i][4] = str(run[i][4])
        return run

    system_answers = []
    system_ranks = []
    for i in range(len(predictions)):
        data, output = predictions[i]
        if 'answer' in output:
            sents = to_sentence(data, output['answer'])
            remove_duplicate(sents)

        for i in range(len(data['id'])):
            id = data['id'][i].item()
            if 'answer' in output:
                system_answers.append([';'.join(dataset.context_id(id)), dataset.query_id(id), ';'.join(dataset.passage_id(id)), detokenizer(sents[i])])

            if 'rank' in output:
                scores=output['rank']
                temp = []
                for j in range(len(dataset.pool(id))):
                    temp.append([dataset.query_id(id), 'Q0', dataset.pool(id)[j], 0, scores[i, j].item(), 'system'])
                system_ranks.append(sort(temp))

    output_path = os.path.join(output_path, 'result/')
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if len(system_answers)>0:
        output_path1 = os.path.join(output_path, eval_type+'_'+str(epoch) + '.' + str(local_rank) + '.answer')
        file = codecs.open(output_path1, "w", "utf-8")
        for i in range(len(system_answers)):
            file.write('\t'.join(system_answers[i]) + os.linesep)
        file.close()
    if len(system_ranks) > 0:
        output_path2 = os.path.join(output_path, eval_type+'_'+str(epoch) + '.' + str(local_rank) + '.run')
        file = codecs.open(output_path2, "w", "utf-8")
        for i in range(len(system_ranks)):
            for j in range(len(system_ranks[i])):
                file.write(' '.join(system_ranks[i][j]) + os.linesep)
        file.close()