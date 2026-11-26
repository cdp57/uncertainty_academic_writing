#!/usr/bin/env python3
"""CPU-only regression test for Qwen answer parsing."""
from demo4_prompts import parse_answer, proposed_answer, confidence_from_raw

cases = [
    ('ANSWER: <answer>oxidants</answer>', 'oxidants'),
    ('ANSWER: <answer>clone</answer>', 'clone'),
    ('<answer>reductants</answer>', 'reductants'),
    ('<answer>Unanswerable</answer>', 'Unanswerable'),
    ('<answer> somatic cell </answer>', 'somatic cell'),
    ('ANSWER: <answer> pivot-based method and multilingual NMT (MNMT) </answer>',
     'pivot-based method and multilingual NMT (MNMT)'),
    ('ANSWER: <answer>Pivot-based methods include pivoting and pivot-synthetic approaches, '
     'where the pivot language is used as a bridge for translation between source and target languages.',
     'Pivot-based methods include pivoting and pivot-synthetic approaches, where the pivot language '
     'is used as a bridge for translation between source and target languages'),
    ('ANSWER: Yeast', 'Yeast'),
    ('Mesophilic organisms', 'Mesophilic organisms'),
    ('ANSWER: <answer>', None),
]
for raw, expected in cases:
    parsed = parse_answer(raw)
    assert parsed == expected, f'{raw!r}: expected {expected!r}, got {parsed!r}'
    if expected:
        assert proposed_answer(None, raw) == expected
assert confidence_from_raw('CONFIDENCE: 8') == 8
print(f'PASS: all {len(cases)} answer parsing cases and confidence parsing.')
