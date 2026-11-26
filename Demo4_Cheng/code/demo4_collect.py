#!/usr/bin/env python3
"""Serial single-GPU Demo 4 collector, using frozen Demo 3 inputs.

Per (model, condition, question, method): EXACTLY two generation calls, one
answer and one confidence evaluation. Stage 1 is checkpointed BEFORE stage 2,
so interrupted jobs never have to regenerate successful answer stages.

Method availability: structured_rerun = Qwen3 1.7B / Qwen3 4B only;
academic_system = all four models. One sampled output for each stage.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

from demo3_data import COUNTS
from demo3_prepare import DATA_DIR, validate_frozen
from demo4_prompts import (
    PROMPT_VERSION, METHODS, system_prompt, structured_answer_prompt,
    structured_confidence_prompt, parse_answer, confidence_from_raw, proposed_answer,
)

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'results' / 'demo4' / 'responses'
GLOBAL_SEED = 42  # One sample, not five random-seed repetitions.
MODEL_IDS = {
    'llama_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama_3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen_1p7b': 'Qwen/Qwen3-1.7B',
    'qwen_4b': 'Qwen/Qwen3-4B',
}
MODEL_METHODS = {
    'llama_1b': ('academic_system',),
    'llama_3b': ('academic_system',),
    'qwen_1p7b': ('structured_rerun', 'academic_system'),
    'qwen_4b': ('structured_rerun', 'academic_system'),
}
STAGES = ('answer', 'confidence')
STOP_REQUESTED = False


def stop_handler(sig, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'Signal {sig}: stopping after the active generation, with checkpoint saved.', flush=True)


def deterministic_seed(model_key, dataset, idx, stage):
    # Same stage seed across structured_rerun vs academic_system: paired comparison.
    msg = f'{GLOBAL_SEED}:{model_key}:{dataset}:{idx}:{stage}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(msg).digest()[:4], 'big') & 0x7fffffff


def load_inputs(dataset):
    """Read-only validation against the original Demo 3 frozen manifest."""
    sha = validate_frozen(dataset)
    manifest_path = DATA_DIR / 'manifest.json'
    if not manifest_path.exists():
        raise RuntimeError(f'Frozen manifest not found: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get(dataset, {}).get('sha256') != sha:
        raise RuntimeError(f'Demo 3 input SHA mismatch: {dataset}. Do not regenerate inputs.')
    with (DATA_DIR / f'{dataset}.jsonl').open(encoding='utf-8') as fh:
        return [json.loads(line) for line in fh]


def response_file(model, dataset):
    return OUTPUT_DIR / f'{model}_{dataset}.jsonl'


def read_checkpoint(path, model, dataset, *, repair=False):
    """Return {(idx, method): {stage: row}}; optionally trim a torn final line."""
    entries = {}
    if not path.exists():
        return entries
    valid_end = 0
    with path.open('rb') as fh:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if fh.readline():
                    raise ValueError(f'Malformed row before EOF in {path} at byte {pos}')
                print(f'Incomplete trailing JSONL row: {path}', flush=True)
                break
            if (row.get('model_key') != model or row.get('dataset') != dataset
                    or row.get('prompt_version') != PROMPT_VERSION):
                raise ValueError(f'Incompatible checkpoint record at {path}:{pos}')
            method, stage, idx = row.get('method'), row.get('stage'), row.get('global_idx')
            if method not in MODEL_METHODS[model] or stage not in STAGES or type(idx) is not int:
                raise ValueError(f'Invalid method/stage/index at {path}:{pos}')
            pair = entries.setdefault((idx, method), {})
            if stage in pair:
                raise ValueError(f'Duplicate {stage} for {model}/{dataset}/{idx}/{method}')
            if stage == 'confidence' and 'answer' not in pair:
                raise ValueError(f'Confidence stored without answer stage in {path}:{pos}')
            pair[stage] = row
            valid_end = fh.tell()
    if repair and valid_end != path.stat().st_size:
        with path.open('rb+') as fh:
            fh.truncate(valid_end)
    if repair and valid_end:
        with path.open('rb+') as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b'\n':
                fh.write(b'\n')
    return entries


class Inference:
    def __init__(self, model_key, quantization, max_input_tokens):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU detected. Submit using scripts/demo4_*.slurm')
        self.torch = torch
        self.family = 'qwen' if model_key.startswith('qwen') else 'llama'
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS[model_key], local_files_only=True)
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_tokens = max_input_tokens
        params = {'device_map': {'': 0}, 'dtype': torch.float16, 'low_cpu_mem_usage': True,
                  'local_files_only': True}
        if quantization == '4bit':
            params['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_IDS[model_key], **params)
        self.model.eval()
        self.device = torch.device('cuda:0')
        print(f'GPU={torch.cuda.get_device_name(0)}; model={MODEL_IDS[model_key]}; '
              f'quantization={quantization}', flush=True)

    def generate(self, method, stage, user_prompt, seed, token_limit):
        tokenizer = self.tokenizer
        kwargs = {'tokenize': False, 'add_generation_prompt': True}
        if self.family == 'qwen':
            kwargs['enable_thinking'] = False
        text = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system_prompt(method)},
             {'role': 'user', 'content': user_prompt}], **kwargs)
        # Qwen occasionally spends the entire short confidence budget on
        # an explanation.  Prefill the ASSISTANT response with the expected
        # label, just as Demo 3's money-betting response was prefilled.
        # This does NOT change either the user prompt or the system prompt.
        # Both Qwen methods receive exactly the same decoding treatment.
        prefix = 'CONFIDENCE: ' if self.family == 'qwen' and stage == 'confidence' else ''
        text += prefix
        encoded = tokenizer(text, add_special_tokens=False, return_tensors='pt')
        original = encoded['input_ids'].shape[1]
        truncated = original > self.max_input_tokens
        if truncated:
            encoded = {k: v[:, -self.max_input_tokens:] for k, v in encoded.items()}
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        started = time.perf_counter()
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=True, temperature=0.7,
                top_p=0.8 if self.family == 'qwen' else 1.0,
                max_new_tokens=token_limit, num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                use_cache=True)
        elapsed = round(time.perf_counter() - started, 3)
        n = encoded['input_ids'].shape[1]
        completion = tokenizer.decode(generated[0][n:], skip_special_tokens=True).strip()
        raw = prefix + completion
        return raw, n, int(generated.shape[1] - n), elapsed, truncated


def collect(args, *, inference_cls=Inference):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = tuple(COUNTS) if args.dataset == 'all' else (args.dataset,)
    methods = MODEL_METHODS[args.model]
    plan = []
    for dataset in datasets:
        records = load_inputs(dataset)
        if args.limit_per_dataset is not None:
            records = records[:args.limit_per_dataset]
        path = response_file(args.model, dataset)
        checkpoint = read_checkpoint(path, args.model, dataset, repair=True)
        remaining = sum(1 for r in records for m in methods
                        for s in STAGES if s not in checkpoint.get((r['global_idx'], m), {}))
        print(f'{args.model} {dataset}: {len(records)} questions, '
              f'{remaining}/{len(records)*len(methods)*2} generation stages remaining', flush=True)
        if remaining:
            plan.append((dataset, records, path, checkpoint))
    if not plan:
        print('Everything requested already completed: no GPU model loaded.', flush=True)
        return 0
    infer = inference_cls(args.model, args.quantization, args.max_input_tokens)
    new_stages = 0
    start = time.monotonic()
    for dataset, records, path, ckpt in plan:
        with path.open('a', encoding='utf-8', buffering=1) as fh:
            for rec in records:
                idx = rec['global_idx']
                for method in methods:
                    pair = ckpt.setdefault((idx, method), {})
                    for stage in STAGES:
                        if stage in pair:
                            continue
                        if STOP_REQUESTED:
                            print('Graceful stop; rerun same Slurm script to resume.', flush=True)
                            return 3
                        if stage == 'answer':
                            user_prompt = structured_answer_prompt(rec)
                            tokens = args.max_answer_tokens
                        else:
                            first = pair['answer']
                            given = proposed_answer(first.get('answer'), first.get('raw'))
                            user_prompt = structured_confidence_prompt(rec, given)
                            tokens = args.max_confidence_tokens
                        seed = deterministic_seed(args.model, dataset, idx, stage)
                        raw, in_toks, out_toks, sec, truncated = infer.generate(
                            method, stage, user_prompt, seed, tokens)
                        row = {
                            'model_key': args.model, 'model_id': MODEL_IDS[args.model],
                            'dataset': dataset, 'global_idx': idx,
                            'paper_idx': rec.get('paper_idx'),
                            'question_idx': rec.get('question_idx'),
                            'method': method, 'stage': stage,
                            'prompt_version': PROMPT_VERSION,
                            'run_seed': GLOBAL_SEED, 'sampling_seed': seed,
                            'temperature': 0.7,
                            'raw': raw,
                            'answer': parse_answer(raw) if stage == 'answer' else None,
                            'confidence': confidence_from_raw(raw) if stage == 'confidence' else None,
                            'input_tokens': int(in_toks), 'output_tokens': int(out_toks),
                            'elapsed_seconds': sec, 'input_truncated': bool(truncated),
                        }
                        # Save each stage before the next model call; no redundant re-generation.
                        fh.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                        fh.flush()
                        pair[stage] = row
                        new_stages += 1
                        if new_stages % 100 == 0:
                            print(f'Generated stages={new_stages} dataset={dataset} idx={idx} '
                                  f'elapsed={time.monotonic()-start:.0f}s', flush=True)
            print(f'Finished {dataset} for {args.model}', flush=True)
    print(f'Completed {new_stages} new generation stages in {time.monotonic()-start:.1f} seconds.', flush=True)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODEL_IDS)
    parser.add_argument('--dataset', choices=['all', *COUNTS], default='all')
    parser.add_argument('--limit-per-dataset', type=int, default=None,
                        help='Pilot: first N records per dataset, retained for full run')
    parser.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    parser.add_argument('--max-input-tokens', type=int, default=6144)
    parser.add_argument('--max-answer-tokens', type=int, default=128)
    parser.add_argument('--max-confidence-tokens', type=int, default=96)
    parser.add_argument('--validate-only', action='store_true',
                        help='CPU-only: validate frozen inputs, do not load GPU')
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        parser.error('--limit-per-dataset must be >= 1')
    if min(args.max_input_tokens, args.max_answer_tokens, args.max_confidence_tokens) < 1:
        parser.error('Token counts must be positive')
    datasets = tuple(COUNTS) if args.dataset == 'all' else (args.dataset,)
    if args.validate_only:
        for ds in datasets:
            rows = load_inputs(ds)
            print(f'OK: {ds} {len(rows)} prepared records, frozen hash verified')
        print(f'OK: model={args.model}, methods={MODEL_METHODS[args.model]}; NO GPU used')
        return 0
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = OUTPUT_DIR / f'.{args.model}.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as ex:
            raise RuntimeError(f'A {args.model} collector is already running') from ex
        return collect(args)


if __name__ == '__main__':
    sys.exit(main())
