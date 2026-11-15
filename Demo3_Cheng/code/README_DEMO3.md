# Demo 3 — single-GPU serial data collection

This is an **add-on to the existing `~/447_demo2/` project**, not a replacement for its Demo 2 scripts. All new Python code is at the root, all Slurm scripts in `scripts/`, all Slurm `.out`/`.err` in `outputs/`, and all new data beneath `results/demo3/`.

## Exactly what runs

| Dataset condition | Split / input | Questions |
|---|---|---:|
| `sciq` | `allenai/sciq`, **test** | 1000 |
| `sciq_bias_incorrect` | Same 1000 SciQ test rows, matched to `sciq_biased_questions_cleaned.csv`, `biased_incorrect_question` **only** | 1000 |
| `qasper` | `allenai/qasper`, **validation**, flattened QAs from 281 papers | 1005 |

Total = **3005 unique question/condition records per model**. Four models = Llama 3.2 Instruct 1B/3B and Qwen3 1.7B/4B. Each question uses **exactly five new Demo 3 methods** (Speculative Editor, Opposing Counsel, Compliance Officer, Panel of Personas, Money Betting) and **one generated response for each method**, temperature 0.7. There are **no five-seed repetitions, no baseline reruns, no arrays, no multi-GPU jobs, no model or data parallelism, no retries**. Total target: **60,100 method responses**.

A single *root run seed* (`42`) is used to derive per-record deterministic RNG states, so a resumed job does not change a yet-to-be-generated record's sampling. These states are **not replicate experiments**.

Qwen3 uses `enable_thinking=False` and `top_p=0.8`; Llama uses `top_p=1.0`, matching the Demo 2 generation settings. 4-bit NF4 weights, one GPU and one process are used per job. The Money Betting response is prefixed on the assistant side with `BET_USD: `, then the answer follows. No Kelly Criterion text is used.

All prompts use the Demo 2 reference support instruction. SciQ uses its provided `support`. QASPER uses the *same* title+abstract+lexically retrieved paragraphs/captions as Demo 2, capped at 12000 characters, **not** oracle evidence and **not** the entire article. The gold answers are never inserted in model prompts.

## Layout

```text
447_demo2/
  demo3_prepare.py                  # CPU: freeze/validate inputs
  demo3_data.py                     # previous dataset selection, QASPER retrieval
  demo3_prompts.py                  # five new prompts, parsers
  demo3_collect.py                  # serial single-GPU collector, JSONL checkpoint
  demo3_export.py                   # CPU: question-wise wide CSVs
  demo3_status.py                   # CPU: progress
  demo3_prefetch_models.py          # CPU: HF weights download once
  setup_env_cxg438.sh               # existing compatible venv/cache setup
  sciq_biased_questions_cleaned.csv # same aligned 1000-question cue dataset
  README_DEMO3.md
  scripts/
    demo3_llama_1b.slurm
    demo3_llama_3b.slurm
    demo3_qwen_1p7b.slurm
    demo3_qwen_4b.slurm
    submit_demo3_serial.sh
  outputs/                           # .out and .err here
  results/demo3/
    inputs/                          # SciQ/QASPER frozen inputs + manifest
    responses/                       # 12 JSONL checkpoints, lossless raw generations
    csv/                             # exported wide CSV files + coverage
```

## Preparation (login/CPU node: NO GPU)

Copy the ZIP's `447_demo2/` contents **into your existing** `~/447_demo2/` directory; do **not** remove existing Demo 2 code or results. If required, update the cluster account/partition in the `.slurm` files (`-A csds447`, `-p markov_gpu` from the previous project).

```bash
cd ~/447_demo2
bash setup_env_cxg438.sh              # one-time venv/dependencies; if not already done
source setup_env_cxg438.sh            # activate HF course-storage cache variables
source "$VENV_DIR/bin/activate"
python demo3_prepare.py               # 3005 frozen records; downloads datasets once
python demo3_prepare.py --validate-only
python demo3_prefetch_models.py       # download/cache all 4 weights on CPU, serially
python demo3_status.py                # prints 0/60,100 before collection
mkdir -p outputs results/demo3/responses results/demo3/csv
```

Prefetch is deliberately done **outside the GPU job**. Llama models are gated: ensure access is approved and your Hugging Face token has been saved by setup. GPU scripts use `HF_HUB_OFFLINE=1` to avoid paying GPU time for downloads. If the full snapshots are already cached from Demo 2, the prefetch step should mostly reuse them. Respect local login-node download policies.

## Smoke test (ONE GPU, 2 questions from each condition)

```bash
cd ~/447_demo2
sbatch scripts/demo3_llama_1b.slurm --limit-per-dataset 2
squeue -u "$USER"
# After completion: inspect outputs/d3_llama_1b-<jobid>.out/.err
python demo3_status.py
```

A successful smoke test creates **30 responses**: 2 questions x 3 conditions x 5 methods. The full job reuses them; they do not need to be regenerated. The GPU reservation is released when the short pilot completes.

## Full collection (no concurrency)

Recommended: **submit one model at a time** and check its output before submitting the next. Each script processes all three conditions serially within one model load.

```bash
cd ~/447_demo2
sbatch scripts/demo3_llama_1b.slurm
# When DONE: sbatch scripts/demo3_llama_3b.slurm
# When DONE: sbatch scripts/demo3_qwen_1p7b.slurm
# When DONE: sbatch scripts/demo3_qwen_4b.slurm
```

Alternatively, `bash scripts/submit_demo3_serial.sh` enqueues the four model scripts with `afterok` dependencies. At most **one** of these jobs can run at once. **Do not do both**; submitting the same jobs twice can corrupt checkpoints.

**Time limits and resuming:** Each job requests one GPU up to ten hours, but it exits as soon as its remaining responses finish (unused reservation is released). Every method response is immediately appended to JSONL. A nonblocking per-model file lock prevents accidental simultaneous collectors for the same model; status checks never truncate a live checkpoint. `--signal=USR1@240` triggers a graceful stop near the time limit, and the job exits nonzero when incomplete. Resubmit **the same** model job: it skips completed records (including a pilot). If you used the dependency launcher and the job times out, downstream dependent jobs won't launch; finish the failed model and submit the remaining ones manually. If the GPU time budget is tight, inspect speed on the first pilot rather than launching all models blindly.

No accuracy judgments, gold-label matching, extra prompts, or LLM-as-judge run during collection. Incorrectly formatted outputs are preserved as raw text with `None` for unavailable parsed fields, **not retried**.

## CPU-only status/export

```bash
cd ~/447_demo2
python demo3_status.py
python demo3_export.py --allow-incomplete   # optional: inspect pilot
python demo3_export.py                      # validates complete 60,100 set
```

Files under `results/demo3/responses/` contain one JSON object per `(model, dataset, question, method)`, with raw text, parsed answer/confidence/bet, run seed, token counts, and elapsed generation time. Files under `results/demo3/csv/` contain **12 wide CSVs**, one row per question/model/dataset condition and five method columns, plus `collection_coverage.csv`. CSVs include gold answers for later **separate** evaluation; raw model prompts do not.

`python demo3_export.py` refuses incomplete collections by default. `--allow-incomplete` writes partially available data and coverage counts without misrepresenting it as a complete experiment.

## Notes on efficiency and reproducibility

- No multi-GPU or batch generation. No unnecessary inference for Demo 2 baselines, multiple seeds, or LLM judging.
- One model load per job, then SciQ, SciQ biased-incorrect, and QASPER processed sequentially.
- Input and model snapshots prepared on CPU to avoid unnecessary GPU idle time.
- Per-method checkpointing prevents repeating completed generations after interruptions.
- Prompts/outputs are deliberately short. QASPER input text is retrieval-limited just as in Demo 2.
- Project caches/virtual environment stay in `/mnt/vstor/courses/csds447/$USER`, while result CSV/JSONL/logs remain under `~/447_demo2/`.
- Full 60,100-generation throughput may still require **substantial GPU hours**. We have not run these GPU jobs or measured their actual throughput here; this package is prepared and locally tested with synthetic model outputs, not experimentally validated on your cluster.
