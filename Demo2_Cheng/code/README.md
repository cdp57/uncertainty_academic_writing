# CSDS 447 — Demo 2: Cheng's collection pipeline

This project implements **four Cheng-owned methods** from `447_Demo-2.pdf`:

- **Vanilla / standard self-report**: 5 independent sampled `ANSWER + CONFIDENCE` responses, one prompt per sample (implemented in a single `generate(... num_return_sequences=5)` call).
- **Structured self-report**: 5 independently sampled answers; a **separate prompt conditioned on each answer** produces a confidence score (5 paired confidence samples).
- **P(True)**: 5 sampled proposed answers. One further scoring prompt uses all 5 as brainstormed ideas and the *first* as the candidate. The model scores the continuations ` (A)` / ` (B)` and the script **normalizes the log-likelihoods over these two candidates**. `p_true_probability_A + p_true_probability_B = 1`. This is not an ordinary 1–10 confidence, and is not the full distribution over every possible output.
- **Initial caution prompt**: 5 new sampled `ANSWER + CONFIDENCE` responses prompted to avoid overconfidence. These are **independent generations**, not reused vanilla outputs.

**LLM-as-a-Judge is not run in the collection phase**, as specified in the project document. Separate accuracy and AUROC/AURC evaluation will be performed later; the merge summaries are *not* accuracy estimates.

## Scope (updated)

- Llama 3.2 **1B, 3B**; Qwen3 **1.7B, 4B**.
- **SciQ test** = 1000 questions with provided support paragraph. **QASPER validation** = 1005 questions nested inside 281 paper records, flattened to individual questions.
- **Support only**: there are no no-support jobs or prompts.
- **5 sampled responses per baseline per question** using `temperature=0.7`; **3 run seeds** = `42, 123, 2026`.
- 4 question-index shards per seed and model/dataset = **8 SLURM scripts**, each with 12 array tasks (3 seeds × 4 shards). Jobs are resumable by CSV checkpoint.
- Qwen3 hard-disables thinking (`enable_thinking=False`) to align compact outputs.

## Files

```
~/447_demo2/
  llama_sciq.py, llama_qasper.py, qwen_sciq.py, qwen_qasper.py
  experiment_core.py, merge_results.py, plot_summary.py
  setup_env_cxg438.sh
  scripts/
    llama_1b_sciq.slurm, llama_3b_sciq.slurm
    llama_1b_qasper.slurm, llama_3b_qasper.slurm
    qwen_1p7b_sciq.slurm, qwen_4b_sciq.slurm
    qwen_1p7b_qasper.slurm, qwen_4b_qasper.slurm
  outputs/                     # SLURM .out/.err ONLY, remains in HOME
  results/                     # CSV/XLSX checkpoints, remains in HOME
    merged/                    # generated after completion
```

`~/447_demo2` is in HOME. **Only** the virtual environment, Hugging Face models/datasets, and other caches are stored under `/mnt/vstor/courses/csds447/$USER/{.venv,hf_cache,cache}`.

## 1. Install and authenticate (once on login node)

```bash
cd ~/447_demo2
bash setup_env_cxg438.sh
```

This configures the compatible CUDA 12.8 PyTorch/torchvision packages and asks for the HF token (hidden) if not already saved. You must have permission to access the Meta Llama 3.2 model repos on Hugging Face.

## 2. Small smoke test (GPU, one array task)

The CLI flags after the `.slurm` filename are passed to Python. Example:

```bash
cd ~/447_demo2
sbatch --array=0 scripts/llama_1b_sciq.slurm --num-questions 8
sbatch --array=0 scripts/qwen_1p7b_qasper.slurm --num-questions 8
```

An 8-question pilot and the full run use the same output names. Resubmitting the full job **resumes the pilot** (does not overwrite it).

## 3. Launch all eight experiments (96 array tasks)

```bash
cd ~/447_demo2
for script in scripts/*.slurm; do sbatch "$script"; done
squeue -u "$USER"
```

Each array limits its own concurrent tasks to 2 via `%2`. School account limits may require fewer concurrent jobs; adjust `%2` or submit scripts in stages. Compute time may be substantial because *each question* requires 5+5+5+5 sampled outputs and additional confidence/scoring calls.

### Resuming

CSV checkpoint rows are saved after every five **completed questions**, plus on normal termination or receipt of a stop signal. SLURM `USR1@240` sends a warning signal before its time limit. Rerun the same `.slurm` script to fill missing questions. Every row key `(seed, global_question_idx)` is kept once per shard. Models are not loaded when no work remains.

## 4. Merge completed shards

```bash
source /mnt/vstor/courses/csds447/$USER/.venv/bin/activate
python merge_results.py --results-dir results
```

The merge utility validates **1000 SciQ or 1005 QASPER questions per seed**, rejects duplicated question/seed rows, writes combined CSV/XLSX files under `results/merged/`, and creates per-seed + mean/std across-seed summaries. For an incomplete pilot, use `--allow-incomplete`. The resulting summary includes *confidence statistics*, not judged accuracy.

Optional visualization after merge:

```bash
python plot_summary.py --results-dir results
```

Figures are saved under `results/plots/` with across-seed error bars. They show reported confidence/P(True) scores, **not accuracy**.

## QASPER compatibility fix (October 2026)

The old call `load_dataset("allenai/qasper", "qasper", split="validation")`
fails under current Hugging Face `datasets` with: `Dataset scripts are no longer supported`.
The new `load_qasper_validation()` in `experiment_core.py` downloads
`qasper/validation-00000-of-00001.parquet` at the pinned official snapshot
`13b496d2a5359329b110e3419628de3cf791843b` using `hf_hub_download`
and then loads it using the built-in `parquet` loader.  This preserves the
**281 validation papers / 1005 questions**, and the downloaded data is cached
under `HF_HOME`, in course storage. No package downgrade needed.

To test data loading before requesting a GPU (while logged into a node with
network access and your environment activated):

```bash
cd ~/447_demo2
source setup_env_cxg438.sh
source /mnt/vstor/courses/csds447/$USER/.venv/bin/activate
python qwen_qasper.py --model 1.7B --dataset-only --num-questions 8
```

Old installations may still have `scripts/*.py` from the first package.
Once the new top-level files exist, delete those old helper copies:

```bash
rm -f scripts/experiment_core.py scripts/merge_results.py scripts/plot_summary.py scripts/__init__.py
rm -rf scripts/__pycache__
```

Do **not** delete `results/` or `outputs/`; reruns resume the same CSV files.

## How QASPER support is constructed

QASPER is organized **one row per paper** with nested `qas.question` arrays. We flatten these to 1005 validation questions, preserving `paper_idx` and per-paper `question_idx`, and save multiple annotator references for **later** scoring. The prompt receives the paper title + abstract + deterministic lexical retrieval of selected **paper paragraphs/captions**, limited to 12,000 characters, and not annotator answers or their evidence fields. This is an explicitly **retrieval-limited paper context baseline**, not a claim to have shown the entire paper. Retrieval is identical across all models and seeds.

If you prefer to use full papers rather than retrieved excerpts, that is a **methodology change**, not just a path change: you must set appropriate context windows and GPU memory limits and rerun all model comparisons.

## Important output details

- One CSV row **per question per seed**, with 5 responses/confidences for each method stored as `" || "` text *and* unambiguous JSON arrays, plus raw generations for auditing.
- Column spellings such as `vanilia_baseline_*` and `structured_self-report_*` intentionally follow the provided course PDF.
- P(True) `*_all_probability_A`/`B` fields contain **one** value each because the document says to judge the set of five brainstormed answers **once**.
- Gold references are stored only in the output row, never in model prompts or P(True) scoring.
- Per-question stage-specific RNG seeds are deterministic across SLURM resumptions. Across-seed standard deviation reflects 3 independent seeds, not five duplicate output strings.
- Very long support is token-truncated from the left if needed, retaining the ending with the question and output format. Check `max_input_tokens`/`max_context_chars` for fairness comparisons.

## Validation and caveats

Source code has been syntax-checked and logic-tested with synthetic dataset records and a fake inference implementation. The cluster CUDA driver, RAM capacity, gated Hugging Face downloads, runtime, and actual large-model outputs cannot be verified outside your HPC. Inspect `.out/.err` from the small pilot first. In particular, QASPER answer quality with retrieval-selected excerpts should not be reported as full-document QA, and P(True) is model preference among A/B candidates rather than an empirical correctness probability until calibrated.
