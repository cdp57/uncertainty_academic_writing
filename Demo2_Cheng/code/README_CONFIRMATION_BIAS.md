# SciQ Confirmation-Bias Extension — Cheng / Demo 2

This add-on **does not modify** the original SciQ or QASPER experiments. Place its three Python files in `~/447_demo2/`, and its eight SLURM scripts in `~/447_demo2/scripts/`. Retain your existing `experiment_core.py`, `setup_env_cxg438.sh`, and `sciq_biased_questions_cleaned.csv` in the root. Run from `~/447_demo2` so stdout/stderr go to `outputs/`.

## Experimental unit and count

The teammate CSV has **1,000 original SciQ test items** and two text variations for each item:

- `biased_correct_question` → `biased_correct`
- `biased_incorrect_question` → `biased_incorrect`

**2,000 biased cases per model**, each with **four baselines × five individually seeded answers** (`42, 123, 2026, 3407, 5678`, temperature 0.7). With four models: **8,000 cases, 160,000 saved answer samples**. P(True) computes **one** A/B probability score from its five brainstormed answers per case.

Both bias conditions use the **original SciQ support** (reloaded from `allenai/sciq`, test split) and **original gold answer**. We check CSV item indices, original question wording, and gold-answer alignment before any GPU work. The support is **never saved** to results files. The model does not see the reference answer separately from the teammate's biased wording.

The biased questions are often yes/no suggestions. To make original-SciQ exact-match/F1 scores interpretable, a **shared output instruction** after both versions asks for the correct scientific term/entity, not simply "yes" or "no". `biased_question` in saved results preserves the original teammate wording. Because biased wording may change semantics, results are **reference-based proxies**, not guaranteed semantic correctness. Inspect raw answers and representative errors.

## Prerequisites

- Existing working `~/447_demo2/experiment_core.py` must be the **five-distinct-seed version** containing `SAMPLE_SEEDS = (42, 123, 2026, 3407, 5678)` and `Inference`, `flatten_questions`, and `method_results`.
- Existing `~/447_demo2/setup_env_cxg438.sh`, course-storage `.venv`, and Hugging Face auth/cache.
- The teammate CSV `~/447_demo2/sciq_biased_questions_cleaned.csv` unchanged.
- SLURM account `csds447` and `markov_gpu` partition.

## 0. Check input (CPU, no GPU)

```bash
cd ~/447_demo2
module load Python/3.12.3-GCCcore-13.3.0
module load bzip2/1.0.8-GCCcore-13.3.0
source /mnt/vstor/courses/csds447/$USER/.venv/bin/activate
source ./setup_env_cxg438.sh
python collect_bias_sciq.py --family llama --model 1B --dataset-only
```

Expect **1,000 source questions / 2,000 variations**, and no mismatched gold/question indices.

## 1. Collection first

Pilot: `sbatch --array=0 scripts/bias_collect_llama_1b.slurm --num-questions 32`. Because shard 0 receives only indices divisible by 16, this small pilot processes **two original questions, four biased variations** (indices 0 and 16). Test each model family before full collection.

Full collection:

```bash
cd ~/447_demo2
mkdir -p outputs results/confirmation_bias
for script in scripts/bias_collect_*.slurm; do sbatch "$script"; done
```

Each of **four model arrays** has `--array=0-15%2` (16 shards, at most 2 simultaneous tasks per array). A shard handles all four baselines and both conditions for its assigned original indices (`item_idx % 16`). **Each case is generated once**, with five independently seeded samples per baseline. Checkpointing is per bias case; resubmit a failed shard with the same array ID to resume.

After all **64 collection tasks** are `COMPLETED`, merge and verify all data on the login node (no GPU):

```bash
python summarize_bias_sciq.py --phase collection --results-dir results
```

This **refuses to merge** if any model lacks exactly 2,000 unique biased cases.

## 2. Then evaluate (reference metrics + each model judges itself)

Pilot: `sbatch --array=0 scripts/bias_eval_llama_1b.slurm --num-questions 32` after that shard’s matching 32-question-limit collection pilot completes. This judges **four biased variations × 20 saved answers = 80 answers**.

Full evaluation:

```bash
for script in scripts/bias_eval_*.slurm; do sbatch "$script"; done
```

Each evaluator requires its matching completed collection shard and runs:

1. **Reference metrics (CPU):** normalized SciQ exact match and answer token F1 against the **original** gold answer for all four baselines and five samples.
2. **Self-judgment (GPU):** the **same model checkpoint** judges its own generated answer TRUE/FALSE with confidence 1–10. The judge sees the original question, biased wording, reconstructed original support, and candidate answer, **not** the gold answer. Parse failures are tracked separately.
3. **Confirmation-bias indicators:** whether the generated response contains the suggested answer, including the incorrectly suggested term. These are lexical mention rates, not proof of endorsement or correctness.

Expect **40,000 metric rows and 40,000 judge rows per model**, i.e., **160,000 of each** across four models. Each evaluation shard checkpoints judgments every 20 answers and resumes after interruption. Full self-judging may be computationally expensive; monitor GPU time limits.

After all **64 evaluation tasks** complete, run:

```bash
python summarize_bias_sciq.py --phase evaluation --results-dir results
```

## Output locations

```text
~/447_demo2/
  collect_bias_sciq.py
  evaluate_bias_sciq.py
  summarize_bias_sciq.py
  sciq_biased_questions_cleaned.csv               (existing)
  scripts/bias_collect_*.slurm                     (4)
  scripts/bias_eval_*.slurm                        (4)
  outputs/bias_*.out and outputs/bias_*.err        (SLURM logs)
  results/confirmation_bias/
    collection/                                   (4 models × 16 CSV shards)
    merged/                                       (model-level combined collection,
                                                   metric and judge CSVs)
    evaluation/metrics/                           (reference metrics CSV shards)
    evaluation/judgments/                         (self-judge CSV shards)
    summary/                                      (summary CSV and XLSX)
```

The virtual environment, downloaded SciQ dataset, and Hugging Face caches stay under `/mnt/vstor/courses/csds447/$USER/`.

## Interpret results carefully

- Both biased conditions should be compared **within the same original question, baseline, and sample seed**. `paired_bias_effect.csv` contains the difference *incorrect-cue minus correct-cue*; negative means the incorrect cue lowered reference-match accuracy.
- Exact-match and token F1 measure reference-wording agreement; a correct paraphrase may score lower, and binary answers are not directly comparable to SciQ term labels.
- The self-judge's TRUE rate is a **self-assessment**, not independent correctness. Judge parse coverage is separately reported.
- Mentioning the incorrectly suggested answer is a useful behavioral indicator, but can occur when *rejecting* it (e.g., "not antioxidants"). Inspect examples before claiming confirmation-bias adoption.
- The five seeds give **five answer samples per case**, not five independent full-dataset repetitions.
- The caution, P(True), and structured baselines each generate answers independently; P(True)'s A/B probability is one score per case for its first brainstormed answer.
