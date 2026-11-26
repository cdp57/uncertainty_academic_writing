#!/usr/bin/env python3
"""CPU-only dry-run of Demo 4 prompt identity, checkpointing, and resume.

Uses dummy inference. No model download, datasets, Slurm, or GPU needed.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import demo4_collect as co
from demo4_prompts import (
    BASE_SYSTEM_PROMPT, ACADEMIC_SYSTEM_PROMPT,
    structured_answer_prompt, structured_confidence_prompt,
    parse_answer, confidence_from_raw, system_prompt,
)


class FakeInference:
    calls = []
    def __init__(self, model, quantization, max_input_tokens):
        self.model = model

    def generate(self, method, stage, user_prompt, seed, token_limit):
        self.calls.append((self.model, method, stage, user_prompt, seed))
        if stage == 'answer':
            assert user_prompt.endswith('Give a concise answer. Return exactly:\nANSWER: <answer>')
            return 'ANSWER: Mesophilic organisms', 50, 8, 0.01, False
        assert 'PROPOSED ANSWER:\nMesophilic organisms' in user_prompt
        assert 'Now independently assess confidence that this answer is correct' in user_prompt
        return 'CONFIDENCE: 7', 80, 6, 0.01, False


def test():
    assert system_prompt('structured_rerun') == BASE_SYSTEM_PROMPT
    assert system_prompt('academic_system') == ACADEMIC_SYSTEM_PROMPT
    assert parse_answer('ANSWER: Yeast') == 'Yeast'
    assert parse_answer('Mesophilic organisms') == 'Mesophilic organisms'
    assert parse_answer('ANSWER: <answer>') is None
    assert confidence_from_raw('CONFIDENCE: 8') == 8
    assert confidence_from_raw('8/10') == 8
    assert confidence_from_raw('confidence: 12') is None

    orig_dir = co.OUTPUT_DIR
    orig_load = co.load_inputs
    with tempfile.TemporaryDirectory() as td:
        co.OUTPUT_DIR = Path(td)
        co.load_inputs = lambda dataset: [
            {'dataset': dataset, 'global_idx': i, 'question_idx': i, 'question': f'Test {i}?',
             'support': 'Sample context', 'paper_idx': None} for i in range(2)]
        args = SimpleNamespace(model='qwen_1p7b', dataset='all', limit_per_dataset=2,
                               quantization='4bit', max_input_tokens=6144,
                               max_answer_tokens=128, max_confidence_tokens=32)
        try:
            FakeInference.calls = []
            assert co.collect(args, inference_cls=FakeInference) == 0
            assert len(FakeInference.calls) == 2 * 3 * 2 * 2
            # Same first-stage user prompt under both methods, and same RNG seed.
            v = [call for call in FakeInference.calls if call[2] == 'answer']
            for i in range(0, len(v), 2):
                assert v[i][3] == v[i+1][3]
                assert v[i][4] == v[i+1][4]
            for ds in co.COUNTS:
                cp = co.read_checkpoint(co.response_file(args.model, ds), args.model, ds)
                assert len(cp) == 4
                assert all(set(stages) == {'answer','confidence'} for stages in cp.values())
            FakeInference.calls = []
            assert co.collect(args, inference_cls=FakeInference) == 0
            assert len(FakeInference.calls) == 0
            # Partial stage: strip final confidence line and check it gets regenerated only.
            path = co.response_file(args.model, 'sciq')
            lines = path.read_text().splitlines()
            import json
            chosen = next(i for i, s in enumerate(lines)
                          if (json.loads(s)['method'] == 'academic_system' and
                              json.loads(s)['stage'] == 'confidence'))
            path.write_text('\n'.join(lines[:chosen] + lines[chosen+1:]) + '\n')
            # To keep chronological integrity the row must be removed; ordering is key by question,
            # and a later stage may follow. Existing parser permits incomplete pairs.
            FakeInference.calls = []
            args.dataset = 'sciq'
            assert co.collect(args, inference_cls=FakeInference) == 0
            assert len(FakeInference.calls) == 1
            assert FakeInference.calls[0][2] == 'confidence'
            # Torn trailing line should be repaired without losing earlier stages.
            with path.open('ab') as fh:
                fh.write(b'{"bad":')
            assert len(co.read_checkpoint(path, args.model, 'sciq', repair=True)) == 4
        finally:
            co.OUTPUT_DIR = orig_dir
            co.load_inputs = orig_load

    print('Demo 4 CPU self-test PASSED. No GPU used.')


if __name__ == '__main__':
    test()
