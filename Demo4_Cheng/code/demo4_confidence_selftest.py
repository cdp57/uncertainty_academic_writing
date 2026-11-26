#!/usr/bin/env python3
"""Regression tests for Demo 4 confidence extraction. No GPU needed."""
from demo4_prompts import confidence_from_raw, parse_answer

cases = [
    ('CONFIDENCE: 8', 8),
    ('confidence is 8', 8),
    ('<confidence>8</confidence>', 8),
    ('CONFIDENCE: <integer>8</integer>', 8),
    ('CONFIDENCE: <integer 1-10>8</integer>', 8),
    ('CONFIDENCE: <integer 1-10>', None),
    ('"confidence": 7', 7),
    ('{"confidence_tier": 9}', 9),
    ('8/10', 8),
    ('8 out of 10', 8),
    ('8', 8),
    ('Confidence level: 8/10', 8),
    ('Confidence score: 8', 8),
    ('I am 80% confident', 8),
    ('Confidence: 80%', 8),
    ('I think it is 7/10 confidence', 7),
    ('Confidence: 0.8', None),
    ('ANSWER: oxidation 3; my confidence is 8', 8),
    ('The answer is 8; unclear confidence', None),
]
for raw, expected in cases:
    actual = confidence_from_raw(raw)
    assert actual == expected, f'{raw!r}: expected {expected}, got {actual}'
assert parse_answer('ANSWER: <answer>oxidants</answer>') == 'oxidants'
print(f'PASS: {len(cases)} confidence cases and Qwen answer parser')
