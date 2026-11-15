#!/usr/bin/env python3
"""CPU-only: progress by model, condition and method; no dataset download."""
from demo3_collect import MODEL_IDS, read_checkpoint, response_file
from demo3_data import COUNTS
from demo3_prompts import METHODS


def main():
    total = 0
    for model in MODEL_IDS:
        for dataset, questions in COUNTS.items():
            path = response_file(model, dataset)
            keys = read_checkpoint(path, model, dataset)
            counts = [sum((i, m) in keys for i in range(questions)) for m in METHODS]
            total += len(keys)
            print(f'{model:12} {dataset:20} {len(keys):5}/{questions*5:5} | '
                  + '  '.join(f'{m}={n}' for m, n in zip(METHODS, counts)))
    print(f'TOTAL: {total:,}/60,100 method responses completed')


if __name__ == '__main__':
    main()
