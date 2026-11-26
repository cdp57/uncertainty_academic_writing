#!/usr/bin/env python3
"""CPU-only progress report for two-stage Demo 4 collection."""
from demo3_data import COUNTS
from demo4_collect import MODEL_IDS, MODEL_METHODS, STAGES, response_file, read_checkpoint


def main():
    total = 0
    expected = 0
    for model in MODEL_IDS:
        for dataset, n in COUNTS.items():
            methods = MODEL_METHODS[model]
            cp = read_checkpoint(response_file(model, dataset), model, dataset)
            finished = sum('confidence' in cp.get((i, m), {})
                           for i in range(n) for m in methods)
            partial = sum(('answer' in cp.get((i, m), {})) and
                          ('confidence' not in cp.get((i, m), {}))
                          for i in range(n) for m in methods)
            total += finished
            expected += n * len(methods)
            print(f'{model:12} {dataset:20} completed={finished:5}/{n*len(methods):5} '
                  f'answer-only-pending={partial:4}')
    print(f'TOTAL: {total:,}/{expected:,} completed method responses '
          f'(two generation stages each)')


if __name__ == '__main__':
    main()
