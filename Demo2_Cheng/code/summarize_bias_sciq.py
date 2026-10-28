#!/usr/bin/env python3
"""Merge and summarize 2,000 SciQ biased-question cases per model.

Run --phase collection after all collection array jobs finish.
Run --phase evaluation after all evaluation array jobs finish.
Never uses GPUs; all output is under results/confirmation_bias/.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

MODELS=[('llama','1B'),('llama','3B'),('qwen','1.7B'),('qwen','4B')]
CONDITIONS=['biased_correct','biased_incorrect']
METHODS=['vanilla','structured','p_true','caution']


def read(path):
    if not path.exists(): raise FileNotFoundError(f'Missing {path}; a shard may be incomplete')
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        print(f'No rows for {path}',flush=True);return
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp,path)


def avg(vals):
    return sum(vals)/len(vals) if vals else None


def fnum(text):
    try: return float(text)
    except (TypeError,ValueError): return None


def identity(family,model):
    return f'{family}_{model.lower().replace(".","p")}_sciq_bias'


def merge_collection(results, num_shards, count):
    merged_dir=results/'confirmation_bias'/'merged'
    outputs=[]
    for fam, model in MODELS:
        tag=identity(fam,model)
        rows=[];sha=set()
        for shard in range(num_shards):
            path=results/'confirmation_bias'/'collection'/f'{tag}_shard{shard}of{num_shards}.csv'
            loaded=read(path)
            rows+=loaded
            sha.update(r.get('bias_csv_sha256','') for r in loaded)
        keys=[(int(r['global_question_idx']),r['bias_condition']) for r in rows]
        expected={(i,cond) for i in range(count) for cond in CONDITIONS}
        if set(keys)!=expected or len(keys)!=len(expected):
            raise ValueError(f'Incomplete {tag}: {len(set(keys))} unique, expected {len(expected)}; duplicates possible')
        if len(sha)!=1:
            raise ValueError(f'{tag}: CSV source checksums disagree')
        rows.sort(key=lambda x:(int(x['global_question_idx']),x['bias_condition']))
        out=merged_dir/f'{tag}_combined.csv'
        write(out,rows)
        outputs.append({'model':tag,'original_questions':count,'biased_cases':len(rows),
                        'baselines':4,'samples_per_baseline':5,'source_sha256':next(iter(sha)),
                        'merged_file':str(out)})
        print(f'Merged {tag}: {len(rows)} biased-question cases',flush=True)
    return outputs


def summarize_evaluation(results, num_shards, count):
    merged_dir=results/'confirmation_bias'/'merged'
    summaries=[];pairs=[];statuses=[]
    for fam, model in MODELS:
        tag=identity(fam,model)
        metrics=[]; judgments=[]
        for shard in range(num_shards):
            shard_tag=f'{tag}_shard{shard}of{num_shards}.csv'
            metrics+=read(results/'confirmation_bias'/'evaluation'/'metrics'/shard_tag)
            judgments+=read(results/'confirmation_bias'/'evaluation'/'judgments'/shard_tag)
        expect=count*2*4*5
        if len(metrics)!=expect or len(judgments)!=expect:
            raise ValueError(f'{tag} expected {expect} metric/judge records; got {len(metrics)}/{len(judgments)}')
        def k(row):return (int(row['global_question_idx']),row['bias_condition'],row['baseline'],int(row['sample_index']))
        mkeys={k(r) for r in metrics}; jkeys={k(r) for r in judgments}
        if len(mkeys)!=expect or len(jkeys)!=expect or mkeys!=jkeys:
            raise ValueError(f'{tag} duplicate or mismatched question/condition/baseline/sample keys')
        judge_map={k(r):r for r in judgments}
        metrics.sort(key=k)
        judgments.sort(key=k)
        write(merged_dir/f'{tag}_metrics_combined.csv', metrics)
        write(merged_dir/f'{tag}_judgments_combined.csv', judgments)
        statuses.append({'model':tag,'original_questions':count,'biased_cases':count*2,
                         'answer_metric_rows':len(metrics),'judge_rows':len(judgments)})
        groups=defaultdict(list)
        for r in metrics:
            j=judge_map[k(r)]
            groups[(r['bias_condition'],r['baseline'])].append((r,j))
        for cond in CONDITIONS:
            for method in METHODS:
                records=groups[(cond,method)]
                em=[float(r['exact_match']) for r,_ in records]
                f1=[float(r['token_f1']) for r,_ in records]
                valid=[(r,j) for r,j in records if j['judge_status']=='ok']
                true=[1.0 if j['judge_verdict']=='TRUE' else 0.0 for _,j in valid]
                agree=[int((j['judge_verdict']=='TRUE') == (float(r['exact_match'])==1))
                       for r,j in valid]
                confidence=[fnum(j['judge_confidence']) for _,j in valid]
                confidence=[x for x in confidence if x is not None]
                summaries.append({
                    'model':tag,'condition':cond,'baseline':method,'n_answers':len(records),
                    'exact_match_accuracy':avg(em),'token_f1':avg(f1),
                    'self_judge_true_rate_valid_only':avg(true),
                    'self_judge_agreement_with_exact_valid_only':avg(agree),
                    'judge_parse_rate':len(valid)/len(records),
                    'mean_judge_confidence_valid_only':avg(confidence),
                    'mentions_bias_candidate_rate':avg([float(r['mentions_bias_candidate']) for r,_ in records]),
                    'mentions_incorrect_suggestion_rate':avg([float(r['mentions_incorrect_suggestion']) for r,_ in records]),
                })
        for method in METHODS:
            by_key={(int(r['global_question_idx']),r['bias_condition'],int(r['sample_index'])):r
                    for r in metrics if r['baseline']==method}
            p=[]
            for q in range(count):
                for slot in range(1,6):
                    a=by_key[(q,'biased_correct',slot)]
                    b=by_key[(q,'biased_incorrect',slot)]
                    p.append((a,b))
            pairs.append({
                'model':tag,'baseline':method,'paired_samples':len(p),
                'correct_cue_exact_match':avg([float(a['exact_match']) for a,b in p]),
                'incorrect_cue_exact_match':avg([float(b['exact_match']) for a,b in p]),
                'accuracy_drop_incorrect_minus_correct':avg([float(b['exact_match'])-float(a['exact_match']) for a,b in p]),
                'correct_cue_token_f1':avg([float(a['token_f1']) for a,b in p]),
                'incorrect_cue_token_f1':avg([float(b['token_f1']) for a,b in p]),
                'f1_drop_incorrect_minus_correct':avg([float(b['token_f1'])-float(a['token_f1']) for a,b in p]),
                'incorrect_cue_wrong_suggestion_mention_rate':avg([float(b['mentions_incorrect_suggestion']) for a,b in p]),
            })
        print(f'Evaluated and merged {tag}: {expect} reference metrics and judgments',flush=True)
    return summaries,pairs,statuses


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    p.add_argument('--phase',choices=['collection','evaluation','all'],default='all')
    p.add_argument('--num-shards',type=int,default=16)
    p.add_argument('--num-questions',type=int,default=1000,
                   help='For pilot summaries only; default expects all 1000 original questions')
    a=p.parse_args()
    if not 1<=a.num_questions<=1000 or a.num_shards<=0: p.error('Invalid count/shard setting')
    summ_dir=a.results_dir/'confirmation_bias'/'summary'
    if a.phase in ('collection','all'):
        write(summ_dir/'collection_status.csv',merge_collection(a.results_dir,a.num_shards,a.num_questions))
    if a.phase in ('evaluation','all'):
        summary,paired,status=summarize_evaluation(a.results_dir,a.num_shards,a.num_questions)
        write(summ_dir/'evaluation_summary.csv',summary)
        write(summ_dir/'paired_bias_effect.csv',paired)
        write(summ_dir/'evaluation_status.csv',status)
        try:
            import pandas as pd
            with pd.ExcelWriter(summ_dir/'bias_evaluation_summary.xlsx',engine='openpyxl') as xl:
                pd.DataFrame(summary).to_excel(xl,sheet_name='Baseline_by_condition',index=False)
                pd.DataFrame(paired).to_excel(xl,sheet_name='Paired_effects',index=False)
                pd.DataFrame(status).to_excel(xl,sheet_name='Completion',index=False)
        except ImportError:
            print('pandas/openpyxl missing; CSV summaries were saved (Excel skipped)',flush=True)
    print('COMPLETE; summary folder:',summ_dir,flush=True)

if __name__=='__main__':main()
