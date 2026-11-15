#!/usr/bin/env python3
"""Demo 3 input loading. Mirrors the previous Demo 2 SciQ/QASPER selection.

SciQ test: 1000; SciQ biased *incorrect* cue: 1000; QASPER validation:
1005 flattened questions from 281 paper records. References never enter prompts.
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

QASPER_PARQUET_REVISION = '13b496d2a5359329b110e3419628de3cf791843b'
QASPER_PARQUET_FILE = 'qasper/validation-00000-of-00001.parquet'
BIAS_CSV_NAME = 'sciq_biased_questions_cleaned.csv'
COUNTS = {'sciq': 1000, 'sciq_bias_incorrect': 1000, 'qasper': 1005}


def _norm(s):
    return ' '.join(re.findall(r'[a-z0-9]+', str(s).lower()))


def load_qasper_validation():
    """Pinned Parquet snapshot used in Demo 2 (old HF script is obsolete)."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id='allenai/qasper', repo_type='dataset',
        filename=QASPER_PARQUET_FILE, revision=QASPER_PARQUET_REVISION)
    return load_dataset('parquet', data_files={'validation': str(path)}, split='validation')


def _paper_paragraphs(paper):
    text = paper.get('full_text') or {}
    if isinstance(text, dict):
        sections = text.get('section_name') or []
        paragraphs = text.get('paragraphs') or []
        pairs = zip(sections, paragraphs)
    elif isinstance(text, list):
        pairs = ((section.get('section_name', ''), section.get('paragraphs', []))
                 for section in text if isinstance(section, dict))
    else:
        raise TypeError(f'Unexpected QASPER full_text type: {type(text).__name__}')
    out = []
    for name, group in pairs:
        for paragraph in (group if isinstance(group, list) else [group]):
            val = str(paragraph).strip()
            if val:
                out.append((str(name), val))
    figures = paper.get('figures_and_tables') or {}
    if isinstance(figures, dict):
        captions = figures.get('caption') or []
    elif isinstance(figures, list):
        captions = [r.get('caption') for r in figures if isinstance(r, dict)]
    else:
        captions = []
    for caption in captions:
        if caption:
            out.append(('Figure or table caption', str(caption)))
    return out


def retrieve_paper_context(question, title, abstract, paragraphs, budget):
    """Exact Demo 2 lexical retrieval: title + abstract + ranked paper text.

    No annotator evidence/answers are consulted. This is retrieval-limited QA.
    """
    prefix = f'Paper title: {title}\nAbstract: {abstract}\n'
    prefix = prefix[:max(0, budget // 3)]
    query_words = set(re.findall(r'[a-z0-9]{3,}', question.lower()))
    ranked = []
    for i, (section, para) in enumerate(paragraphs):
        pwords = set(re.findall(r'[a-z0-9]{3,}', para.lower()))
        shared = len(query_words & pwords)
        bonus = sum(1 for word in query_words if word in section.lower())
        ranked.append((shared + 2 * bonus, i, section, para))
    ranked.sort(key=lambda r: (-r[0], r[1]))
    remaining = max(0, budget - len(prefix) - 2)
    selected = []
    for _, i, section, para in ranked:
        snippet = f'[{section}] {para}\n'
        if len(snippet) <= remaining:
            selected.append((i, snippet))
            remaining -= len(snippet)
        elif not selected and remaining > 100:
            selected.append((i, snippet[:remaining]))
            remaining = 0
        if remaining < 100:
            break
    selected.sort(key=lambda x: x[0])
    return (prefix + '\n' + ''.join(text for _, text in selected)).strip()


def extract_qasper_answers(answer_group):
    answer_list = answer_group.get('answer', []) if isinstance(answer_group, dict) else answer_group
    if isinstance(answer_list, list):
        unwrapped = []
        for item in answer_list:
            if isinstance(item, dict) and 'answer' in item:
                nested = item['answer']
                unwrapped.extend(nested if isinstance(nested, list) else [nested])
            else:
                unwrapped.append(item)
        answer_list = unwrapped
    if isinstance(answer_list, dict):
        keys = list(answer_list)
        length = max((len(v) if isinstance(v, list) else 1 for v in answer_list.values()), default=0)
        answer_list = [
            {k: (answer_list[k][i] if isinstance(answer_list[k], list) and i < len(answer_list[k])
                 else answer_list[k] if not isinstance(answer_list[k], list) else None)
             for k in keys}
            for i in range(length)
        ]
    if not isinstance(answer_list, list):
        answer_list = [answer_list]
    result = []
    for ann in answer_list:
        if not isinstance(ann, dict):
            continue
        if ann.get('unanswerable') is True:
            value = 'Unanswerable'
        elif (ann.get('free_form_answer') or '').strip():
            value = str(ann['free_form_answer']).strip()
        elif ann.get('extractive_spans'):
            value = ' '.join(str(v) for v in ann['extractive_spans'])
        elif ann.get('yes_no') is True:
            value = 'Yes'
        elif ann.get('yes_no') is False:
            value = 'No'
        else:
            continue
        if value and value not in result:
            result.append(value)
    return result


def _sciq_records():
    from datasets import load_dataset
    dataset = load_dataset('allenai/sciq', split='test')
    if len(dataset) != 1000:
        raise RuntimeError(f'SciQ test expected 1000 questions, found {len(dataset)}')
    records = []
    for idx, ex in enumerate(dataset):
        records.append({
            'dataset': 'sciq', 'global_idx': idx, 'question_idx': idx,
            'paper_idx': None, 'paper_id': '', 'paper_title': '',
            'question': str(ex['question']).strip(),
            'support': str(ex.get('support') or '').strip(),
            'reference_answers': [str(ex['correct_answer']).strip()],
            'context_source': 'SciQ test support field',
            'bias_condition': '', 'original_question': '',
            'incorrect_answer': '',
        })
    return records


def _bias_records(sciq_records, root):
    path = Path(root) / BIAS_CSV_NAME
    if not path.is_file():
        raise FileNotFoundError(f'Missing bias source CSV: {path}')
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh)
        required = {'item_idx', 'original_question', 'correct_answer',
                    'incorrect_answer', 'biased_incorrect_question'}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f'Bias CSV missing fields: {required - set(reader.fieldnames or [])}')
        rows = list(reader)
    if len(rows) != 1000 or {int(r['item_idx']) for r in rows} != set(range(1000)):
        raise ValueError('Bias CSV must contain each SciQ test index 0–999 exactly once')
    result = []
    for r in sorted(rows, key=lambda row: int(row['item_idx'])):
        idx = int(r['item_idx'])
        sciq = sciq_records[idx]
        if (_norm(r['original_question']) != _norm(sciq['question']) or
                _norm(r['correct_answer']) != _norm(sciq['reference_answers'][0])):
            raise ValueError(f'Bias CSV misaligned with SciQ test at index {idx}')
        if not r['biased_incorrect_question'].strip() or not r['incorrect_answer'].strip():
            raise ValueError(f'Blank incorrect cue at index {idx}')
        entry = dict(sciq)
        entry.update({
            'dataset': 'sciq_bias_incorrect', 'bias_condition': 'biased_incorrect',
            'original_question': r['original_question'].strip(),
            'incorrect_answer': r['incorrect_answer'].strip(),
            'question': (
                r['biased_incorrect_question'].strip() +
                '\n\nGive the correct scientific term or entity described by this '
                'question, not just yes or no. If the suggested term is wrong, '
                'give the correct term.'
            ),
            'context_source': 'SciQ test support field plus incorrect-cue wording',
            'bias_csv_sha256': digest,
        })
        result.append(entry)
    return result, digest


def load_records(dataset_name, root, max_context_chars=12000):
    if dataset_name == 'sciq':
        records = _sciq_records()
        digest = 'allenai/sciq:test:1000'
    elif dataset_name == 'sciq_bias_incorrect':
        records, digest = _bias_records(_sciq_records(), root)
    elif dataset_name == 'qasper':
        papers = load_qasper_validation()
        if len(papers) != 281:
            raise RuntimeError(f'Expected 281 QASPER validation papers, found {len(papers)}')
        records = []
        for paper_idx, paper in enumerate(papers):
            qas = paper['qas']
            if isinstance(qas, dict):
                questions = qas.get('question') or []
                annotations = qas.get('answers') or []
            elif isinstance(qas, list):
                questions = [row.get('question', '') for row in qas]
                annotations = [row.get('answers', []) for row in qas]
            else:
                raise TypeError(f'Unexpected QASPER qas type: {type(qas).__name__}')
            paragraphs = _paper_paragraphs(paper)
            for q_idx, question in enumerate(questions):
                selected = retrieve_paper_context(
                    str(question), str(paper.get('title') or ''),
                    str(paper.get('abstract') or ''), paragraphs,
                    max_context_chars)
                answer_group = annotations[q_idx] if q_idx < len(annotations) else {}
                records.append({
                    'dataset': 'qasper', 'paper_idx': paper_idx,
                    'paper_id': str(paper.get('id') or ''),
                    'paper_title': str(paper.get('title') or ''),
                    'question_idx': q_idx, 'global_idx': len(records),
                    'question': str(question).strip(), 'support': selected,
                    'context_source': 'QASPER validation: lexical paper retrieval, no oracle evidence',
                    'reference_answers': extract_qasper_answers(answer_group),
                    'bias_condition': '', 'original_question': '',
                    'incorrect_answer': '',
                })
        digest = f'allenai/qasper:{QASPER_PARQUET_REVISION}:{max_context_chars}'
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    if len(records) != COUNTS[dataset_name]:
        raise RuntimeError(f'{dataset_name}: expected {COUNTS[dataset_name]} questions, found {len(records)}')
    return records, digest
