# Demo 4 — Qwen `<answer>` parsing fix (CPU-first)

**Problem:** Qwen3 generates text such as `ANSWER: <answer>oxidants</answer>` and the original Demo 4 parser treats it as missing. The old confidence call was then shown that entire marked-up string rather than the cleaned answer. This patch changes no experimental prompts or model settings.

## Install

Copy `demo4_prompts.py`, `demo4_repair_checkpoints.py`, and `demo4_parser_selftest.py` to the root of your existing `~/447_demo2/`, keeping your other project files unchanged. The patched `demo4_prompts.py` replaces the old one.

## On the CPU/login node, after the pilot job has ended

```bash
cd ~/447_demo2
python demo4_parser_selftest.py
python demo4_selftest.py
python demo4_repair_checkpoints.py --model qwen_1p7b
# Review the dry-run counts, then:
python demo4_repair_checkpoints.py --model qwen_1p7b --apply
python demo4_export.py --allow-incomplete
```

The repair creates timestamped `.backup-...` copies of the affected JSONL files. It updates parsed answer fields, preserves original raw generations, and removes ONLY confidence-stage records whose stage-2 prompt used the unclean answer string. Repaired answers are kept, so no answer model calls need repeating. The exporter should now report parsed answers for pilot examples and lower parsed confidence counts until the confidence stages are regenerated.

## Resume the two-question pilot

```bash
sbatch scripts/demo4_qwen_1p7b.slurm --limit-per-dataset 2
```

This reloads the model once, then generates only the missing confidence stages (typically 12 on your pilot). After completion:

```bash
python demo4_export.py --allow-incomplete
python - <<'PY'
import pandas as pd
p='results/demo4/csv/collection_coverage.csv'
df=pd.read_csv(p)
df=df[(df.model=='qwen_1p7b') & (df.completed_responses>0)]
for col in ('parsed_answers', 'parsed_scores'):
    df[col+'_rate']=100*df[col]/df.completed_responses
print(df[['dataset','method','completed_responses','parsed_answers_rate','parsed_scores_rate']].to_string(index=False))
PY
```

When pilot coverage is acceptable, you can submit the full serial run as previously planned.

**Safety:** Never run repair during an active Demo 4 job. The utility obtains the same per-model lock used by the collector to prevent concurrent mutation. It does not touch Demo 2, Demo 3, or the frozen inputs.
