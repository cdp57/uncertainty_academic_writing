#!/usr/bin/env python3
"""Single-GPU, single-process, single-sample Demo 3 collector.

Run one model at a time. Each (dataset, question index, method) is generated
exactly once and appended to a resumable JSONL checkpoint. No job arrays,
model parallelism, multi-seed experiments, batch inference, or retries.
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
from demo3_prompts import METHODS, METHOD_TOKEN_LIMITS, PROMPT_VERSION, parse_result, prompt_for

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / 'results' / 'demo3' / 'responses'
GLOBAL_SEED = 42  # One run; method-specific deterministic RNG for resumability.
MODEL_IDS = {
    'llama_1b': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama_3b': 'meta-llama/Llama-3.2-3B-Instruct',
    'qwen_1p7b': 'Qwen/Qwen3-1.7B',
    'qwen_4b': 'Qwen/Qwen3-4B',
}
STOP_REQUESTED = False


def on_stop(signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'Got signal {signum}; will finish and checkpoint current generation.', flush=True)


def method_seed(model_key, dataset, question_idx, method):
    data = f'{GLOBAL_SEED}:{model_key}:{dataset}:{question_idx}:{method}'.encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:4], 'big') & 0x7fffffff


def load_input(dataset):
    sha = validate_frozen(dataset)
    manifest_path = DATA_DIR / 'manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Frozen input manifest missing. Run demo3_prepare.py on CPU first.')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get(dataset, {}).get('sha256') != sha:
        raise RuntimeError(f'Input SHA256 changed for {dataset}; refusing to mix datasets.')
    path = DATA_DIR / f'{dataset}.jsonl'
    with path.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def response_file(model_key, dataset):
    return OUTPUT_DIR / f'{model_key}_{dataset}.jsonl'


def read_checkpoint(path, model_key, dataset, repair=False):
    """Read completed keys, repairing ONLY an incomplete trailing JSONL line."""
    if not path.exists():
        return set()
    valid_end = 0
    keys = set()
    with path.open('rb') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                if f.readline():
                    raise ValueError(f'Malformed checkpoint in middle of {path} at byte {start}')
                print(f'Dropping incomplete trailing JSONL row in {path}', flush=True)
                break
            if row.get('model_key') != model_key or row.get('dataset') != dataset:
                raise ValueError(f'Checkpoint is from a different model/dataset: {path}')
            if row.get('prompt_version') != PROMPT_VERSION:
                raise ValueError(f'Prompt version mismatch in {path}; do not mix experiments.')
            key = (row.get('global_idx'), row.get('method'))
            if key in keys:
                raise ValueError(f'Duplicate checkpoint key {key} in {path}')
            if key[1] not in METHODS or not isinstance(key[0], int):
                raise ValueError(f'Invalid key in {path}: {key}')
            keys.add(key)
            valid_end = f.tell()
    if repair:
        if valid_end < path.stat().st_size:
            with path.open('rb+') as f:
                f.truncate(valid_end)
        # A complete final JSON object can lack the line-ending on interruption.
        # Restore it before appending. Readers never modify live checkpoints.
        if valid_end and path.stat().st_size == valid_end:
            with path.open('rb+') as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b'\n':
                    f.write(b'\n')
    return keys


class Inference:
    def __init__(self, model_key, quantization, max_input_tokens):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        if not torch.cuda.is_available():
            raise RuntimeError('No GPU detected. Launch with sbatch scripts/demo3_*.slurm')
        self.torch = torch
        self.family = 'qwen' if model_key.startswith('qwen') else 'llama'
        model_id = MODEL_IDS[model_key]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.truncation_side = 'left'
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_input_tokens = max_input_tokens
        kwargs = {'device_map': {'': 0}, 'dtype': torch.float16, 'low_cpu_mem_usage': True}
        if quantization == '4bit':
            kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.eval()
        self.device = torch.device('cuda:0')
        print(f'GPU={torch.cuda.get_device_name(0)}; model={model_id}; '
              f'quantization={quantization}', flush=True)

    def format_chat(self, prompt):
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if self.family == 'qwen':
            kwargs['enable_thinking'] = False
        return self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': 'You are a careful scientific question answering assistant. Follow requested formatting.'},
             {'role': 'user', 'content': prompt}], **kwargs)

    def ask(self, method, question, seed):
        import torch
        prompt = prompt_for(method, question)
        formatted = self.format_chat(prompt)
        prefix = 'BET_USD: ' if method == 'money_betting' else ''
        # Prefill is part of the assistant text, not the user prompt.
        formatted += prefix
        encoded = self.tokenizer(formatted, add_special_tokens=False, return_tensors='pt')
        original_length = encoded['input_ids'].shape[1]
        truncated = original_length > self.max_input_tokens
        if truncated:
            encoded = {k: v[:, -self.max_input_tokens:] for k, v in encoded.items()}
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        start = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded, do_sample=True, temperature=0.7,
                top_p=0.8 if self.family == 'qwen' else 1.0,
                max_new_tokens=METHOD_TOKEN_LIMITS[method],
                num_return_sequences=1, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
        n = encoded['input_ids'].shape[1]
        text = self.tokenizer.decode(generated[0][n:], skip_special_tokens=True).strip()
        raw = prefix + text
        return raw, int(n), int(generated.shape[1] - n), round(elapsed, 3), truncated


def collect(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = tuple(COUNTS) if args.dataset == 'all' else (args.dataset,)
    plan = []
    for ds in datasets:
        records = load_input(ds)
        if args.limit_per_dataset is not None:
            records = records[:args.limit_per_dataset]
        path = response_file(args.model, ds)
        done = read_checkpoint(path, args.model, ds, repair=True)
        expected = {(record['global_idx'], method) for record in records for method in METHODS}
        missing_count = len(expected - done)
        print(f'{args.model} {ds}: {len(records)} questions, '
              f'{len(expected) - missing_count}/{len(expected)} method responses complete', flush=True)
        if missing_count:
            plan.append((ds, records, path, done))
    if not plan:
        print('Everything requested is already complete; no GPU model loaded.', flush=True)
        return 0

    infer = Inference(args.model, args.quantization, args.max_input_tokens)
    completed_new = 0
    started = time.monotonic()
    for ds, records, path, done in plan:
        with path.open('a', encoding='utf-8', buffering=1) as fh:
            for record in records:
                for method in METHODS:
                    key = (record['global_idx'], method)
                    if key in done:
                        continue
                    if STOP_REQUESTED:
                        print('Stopping cleanly. Resume by resubmitting the same SLURM job.', flush=True)
                        return 3
                    seed = method_seed(args.model, ds, record['global_idx'], method)
                    raw, input_n, output_n, secs, truncated = infer.ask(method, record, seed)
                    parsed = parse_result(method, raw)
                    result = {
                        'dataset': ds, 'global_idx': record['global_idx'],
                        'paper_idx': record.get('paper_idx'),
                        'question_idx': record.get('question_idx'),
                        'model_key': args.model, 'model_id': MODEL_IDS[args.model],
                        'method': method, 'prompt_version': PROMPT_VERSION,
                        'run_seed': GLOBAL_SEED, 'sampling_seed': seed,
                        'temperature': 0.7, 'raw': raw,
                        'answer': parsed['answer'],
                        'confidence': parsed['confidence'],
                        'bet_usd': parsed['bet_usd'],
                        'panel_json_valid': parsed['panel_json_valid'],
                        'input_tokens': input_n, 'output_tokens': output_n,
                        'elapsed_seconds': secs, 'input_truncated': truncated,
                    }
                    fh.write(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
                    fh.flush()
                    done.add(key)
                    completed_new += 1
                    if completed_new % 50 == 0:
                        print(f'New responses={completed_new}; dataset={ds}; '
                              f'index={record["global_idx"]}; elapsed={time.monotonic()-started:.0f}s',
                              flush=True)
            print(f'Finished {ds} for {args.model}', flush=True)
    print(f'Completed {completed_new} new method responses in '
          f'{time.monotonic()-started:.1f} seconds.', flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=MODEL_IDS)
    ap.add_argument('--dataset', choices=['all', *COUNTS], default='all')
    ap.add_argument('--limit-per-dataset', type=int, default=None,
                    help='Pilot: first N questions per dataset; results retained for full run')
    ap.add_argument('--quantization', choices=['4bit', 'none'], default='4bit')
    ap.add_argument('--max-input-tokens', type=int, default=6144)
    args = ap.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 1:
        ap.error('--limit-per-dataset must be positive')
    if args.max_input_tokens < 128:
        ap.error('--max-input-tokens must be >= 128')
    signal.signal(signal.SIGUSR1, on_stop)
    signal.signal(signal.SIGTERM, on_stop)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Avoid accidental duplicate simultaneous submissions for the same model.
    with (OUTPUT_DIR / f'.{args.model}.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f'{args.model} is already being collected by another job.')
        sys.exit(collect(args))


if __name__ == '__main__':
    main()
