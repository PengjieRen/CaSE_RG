import pytrec_eval
import collections

def eval_trec(run, qrel):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recall','map','ndcg'})
    results = evaluator.evaluate(run)
    return results

def parse_run(f_run):
    run = collections.defaultdict(dict)

    for line in f_run:
        query_id, _, object_id, ranking, score, _ = line.strip().split()

        # assert object_id not in run[query_id]
        run[query_id][object_id] = float(score)

    return run

def eval_trec_file(run_file, ref_file):
    with open(run_file) as f:
        run=parse_run(f)
    with open(ref_file) as f:
        qrel = pytrec_eval.parse_qrel(f)

    results = eval_trec(run, qrel)
    avg=dict()
    for q in results:
        for k in results[q]:
            if k in avg:
                avg[k]+=results[q][k]
            else:
                avg[k]=results[q][k]
    for k in avg:
        avg[k] = avg[k]/len(results)
    return avg