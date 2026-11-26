# Demo 4 — two-stage structured self-report, system-message intervention

An **add-on** to the existing `~/447_demo2/` project. Does **not** replace any Demo 2 or Demo 3 code or results. Requires your *existing* `setup_env_cxg438.sh`, `demo3_data.py`, `demo3_prepare.py`, cached HF model snapshots, and `results/demo3/inputs/{sciq,sciq_bias_incorrect,qasper}.jsonl` + `manifest.json`.

## Exact study design

Three frozen Demo 3 input sets are read **without regenerating them**:

| Dataset condition | Count | Input |
|---|---:|---|
| `sciq` | 1,000 | SciQ **test**, original question and support |
| `sciq_bias_incorrect` | 1,000 | Same SciQ test rows with **incorrect** confirmation-bias cue only |
| `qasper` | 1,005 | QASPER **validation**, Demo 3's retrieved support |
| **Total per applicable model** | **3,005** | |

Models and methods:

| Model | `structured_rerun` (old system prompt) | `academic_system` (new system prompt) | Method responses |
|---|:---:|:---:|---:|
| Llama 3.2 1B Instruct | — | yes | 3,005 |
| Llama 3.2 3B Instruct | — | yes | 3,005 |
| Qwen3 1.7B | yes | yes | 6,010 |
| Qwen3 4B | yes | yes | 6,010 |
| **Total** |  |  | **18,030** |

Each method uses **two distinct model calls** (answer, then confidence about that answer): **36,060 generations total**, one sample per stage, no five-seed repeats. `run_seed=42` derives reproducible per-(model,dataset,question,stage) seeds; for Qwen's two methods the same stage seed is deliberately used so changes come from prompts/sampling trajectories, not different seed choices. The first answer is saved and passed to the second call. The gold reference answer is **never** passed to the model.

**Critical experimental control:** Both methods use the exact same Demo 2 structured **user-prompt templates** in BOTH stages. Only the **system message** differs. The baseline system message is verbatim from Demo 2. The new system message explicitly emphasizes academic Scientific QA, calibrating confidence to the evidence, and avoiding overconfidence. Because each method may generate a different answer at stage 1, the *proposed answer content* in its second user prompt will naturally differ; the stage-2 template is otherwise identical.

Model settings match earlier runs: temperature 0.7, Qwen3 thinking disabled, Qwen `top_p=0.8`, Llama `top_p=1.0`, 4-bit NF4 quantization, 6144 max input tokens, 128 max answer tokens, 32 max confidence tokens. Models load **once per job**, all three datasets are processed serially. **One GPU**, **one process**, **no batch inference**, **no job arrays**, **no parallelism**.

## Files

```text
447_demo2/
  demo4_prompts.py
  demo4_collect.py
  demo4_status.py
  demo4_export.py
  demo4_selftest.py
  README_DEMO4.md
  scripts/
    demo4_llama_1b.slurm
    demo4_llama_3b.slurm
    demo4_qwen_1p7b.slurm
    demo4_qwen_4b.slurm
    submit_demo4_serial.sh
  outputs/                 # Slurm .out/.err
  results/
    demo3/inputs/          # EXISTING, untouched (3005 records + manifest)
    demo4/
      responses/           # 12 JSONL stage checkpoints
      csv/                 # 12 wide CSV exports + collection_coverage.csv
```

## 1. Install files / CPU-only checks (login node, NO GPU)

Unzip this package into the **existing project root**; keep your Demo 2 and 3 files. Then:

```bash
cd ~/447_demo2
module load Python/3.12.3-GCCcore-13.3.0
source setup_env_cxg438.sh
source "$VENV_DIR/bin/activate"

python demo4_selftest.py                   # Fake inference unit test: NO GPU
python demo4_collect.py --model qwen_1p7b --validate-only
python demo4_status.py                     # initially 0/18,030
```

The validation command checks all frozen Demo 3 input hashes and record counts using the old Demo 3 manifest; it does NOT download/recreate datasets, download models, or allocate a GPU. If validation fails, do not submit jobs. **Do not run** `demo3_prepare.py` again unless the frozen inputs truly are missing and you deliberately intend to reconstruct them.

No new environment or model downloads are needed if Demo 3 already ran successfully. The GPU scripts use `HF_HUB_OFFLINE=1` to prevent downloads inside the allocation. If your cluster's Python module is missing `libbz2`, see the prior HPC environment fix; `module load Python/3.12.3-GCCcore-13.3.0` should happen before activating the venv.

## 2. Small GPU pilot

Recommended: pilot the Qwen structured rerun + academic method on two questions per condition:

```bash
cd ~/447_demo2
mkdir -p outputs
sbatch scripts/demo4_qwen_1p7b.slurm --limit-per-dataset 2
squeue --me
```

Pilot expects **12 completed method responses** (2 questions × 3 conditions × 2 methods), i.e. **24 stage generations**. Watch `outputs/d4_qwen_1p7b-<jobid>.out` and `.err` and run `python demo4_status.py`. The full run will skip pilot stages.

## 3. Full serial collection

**Choose ONE submission approach:**

**Preferred automated serial submission, after the pilot ends:**

```bash
cd ~/447_demo2
bash scripts/submit_demo4_serial.sh
```

Creates `afterok` dependencies in the order Qwen 1.7B, Qwen 4B, Llama 1B, Llama 3B. **At most one** runs at a time. Model is loaded once per job. Do not concurrently launch the same script manually.

**Alternatively, manual sequential submission:**

```bash
cd ~/447_demo2
sbatch scripts/demo4_qwen_1p7b.slurm
# Wait for successful completion, then:
sbatch scripts/demo4_qwen_4b.slurm
# Wait, then:
sbatch scripts/demo4_llama_1b.slurm
# Wait, then:
sbatch scripts/demo4_llama_3b.slurm
```

Each job reserves one GPU, up to 10 hours, and frees it immediately if the work finishes early. Jobs process every dataset in series. If a job is interrupted or reaches the time limit, **resubmit that same job**: every saved stage is skipped, including an answer stage completed just before interruption. Do not assume downstream `afterok` jobs will launch if an upstream job fails or times out. Finish and then submit the remaining jobs. Never run both serial-launcher and manual submission simultaneously.

## 4. CPU-only progress and CSV export

```bash
cd ~/447_demo2
python demo4_status.py
python demo4_export.py                   # refuses incomplete results
# For pilot inspection only:
python demo4_export.py --allow-incomplete
```

`results/demo4/csv/` holds 12 model×dataset wide CSVs plus `collection_coverage.csv`. Each row contains question ID, original question, incorrect cue (when applicable), reference answers (for **later CPU evaluation**, not fed to the model), and each method's parsed answer, parsed confidence, both raw generations, token counts, time, and truncation indicators. The baseline `structured_rerun` columns remain empty for Llama because that experiment is intentionally not requested. Parsing failures remain missing rather than fabricated; raw outputs are retained.

**No GPU is used** for `status`, `export`, `selftest`, or the `--validate-only` preflight.

## Resumability and result integrity

- Each model/dataset JSONL file saves TWO events per method-question pair: `stage=answer`, followed by `stage=confidence`.
- Each event is written immediately and flushed. A truncated trailing JSONL line can be repaired on resume; valid earlier stages stay intact.
- Checkpoint keys are unique (`global_idx`, `method`, `stage`); duplicate or mismatched-version data causes an error instead of silently mixing runs.
- A nonblocking per-model lock prevents two simultaneous Demo 4 jobs for the same model.
- The script checks the Demo 3 frozen manifest fingerprint on all three datasets every run.
- Collection deliberately **does not** run an LLM-as-judge, repeated sampling, calibration calculation, or visualization. This keeps collection GPU work limited to the required two calls per method.

## Important comparability note

The new system-prompt intervention aims to influence calibration; it does not guarantee better calibration. To assess it, compare answer accuracy, confidence–accuracy gap, AUROC, AURC, and the biased-minus-unbiased SciQ deltas **on matched question IDs**. As in the Demo 2 comparisons, QASPER exact matching is stricter than semantic correctness. Results are single samples, so do not present them as multiple-seed variance estimates.

This package has been syntax checked and CPU-tested with simulated model outputs. It has **not** run on your HPC or generated experimental results.
