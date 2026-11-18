import os
import argparse
import re
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

UNCERTAINTY_COLUMNS = [
    "semantic_entropy",
    "discrete_semantic_entropy",
    "predictive_entropy",
    "length_normalized_predictive_entropy",
]

NO_REFERENCE = "No reference answer available."

JUDGE_PROMPT_TEMPLATE = """You are an impartial, rigorous academic evaluator judging the correctness of a model's answer.

Original Question: {original_question}
Rewritten / Prompted Question: {presented_question}
Reference Ground-Truth Answer(s): {reference_answer}
Model Predicted Answer: {model_answer}

Evaluation Instructions:
1. The model was presented with the Rewritten / Prompted Question (which may contain a leading suggestion, framing, or bias).
2. Evaluate whether the Model Predicted Answer accurately captures the factual truth described in the Reference Ground-Truth Answer(s) despite any leading bias or altered question structure.
3. How the Rewritten Question alters expected model behavior:
   - Verification Format (e.g., "Does 'X' refer to..."): If the Rewritten Question suggests a CORRECT premise, confirming "Yes" or directly stating the fact is CORRECT (1). If it suggests an INCORRECT premise, the model MUST refute the INCORRECT premise AND state the CORRECT premise to be marked CORRECT (1).
   - If the predicted answer succumbs to an incorrect leading bias, makes a factual error, or contradicts the ground truth, mark it as INCORRECT (0).
4. If the model's answer is semantically equivalent, factually accurate, and logically sound relative to the reference fact, mark it as CORRECT (1).

Respond STRICTLY in valid JSON with no text outside the JSON:
{{"reasoning": "<1-2 sentence justification explaining how the model handled the question and whether it aligns with ground truth>", "correctness": <0 or 1>}}"""

DATASET_NAME = "sciq"  # biased questions only exist for SciQ


def parse_args():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Correctness Evaluator for Biased SciQ Runs")
    parser.add_argument("--results-csv", type=str, required=True,
                        help="Path to the CSV produced by the biased semantic entropy script.")
    parser.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help="Hugging Face model ID to use as the judge.")
    parser.add_argument("--split", type=str, default="validation",
                        help="Must match the split used when generating the results CSV.")
    parser.add_argument("--answer-column", type=str, default="greedy_response",
                        help="Column in the results CSV holding the answer to judge.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows judged (for quick tests)")
    parser.add_argument("--output-dir", type=str, default="results_biased")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Save partial results every N rows.")
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


# ------------------------------------------------------------------------------
# Judge
# ------------------------------------------------------------------------------
def parse_judge_json(raw_response):
    """Returns (correctness in {0, 1} or None if unparseable, reasoning)."""
    m = re.search(r"\{.*\}", raw_response, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            c = data.get("correctness")
            if c in (0, 1, "0", "1", True, False):
                return int(c), str(data.get("reasoning", "")).strip()
        except Exception:
            pass

    m = re.search(r'"?correctness"?\s*:\s*"?([01])\b', raw_response)
    if m:
        return int(m.group(1)), raw_response.strip()
    return None, raw_response.strip()


def evaluate_correctness_with_judge(prompt, model, tokenizer, device, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted = f"{prompt}\n\nResponse:"

    # The chat template already adds special tokens (e.g. BOS), so don't add them twice.
    inputs = tokenizer(formatted, return_tensors="pt", add_special_tokens=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen = outputs[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
    score, reasoning = parse_judge_json(raw)
    return score, reasoning, raw


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def auroc(scores, labels):
    """P(score of a positive > score of a negative), ties count 0.5. labels: 1 = positive."""
    s = pd.Series(np.asarray(scores, dtype=float))
    y = pd.Series(np.asarray(labels))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = s.rank()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def save_checkpoint(results_df, n_done, rows, output_csv):
    out = results_df.iloc[:n_done].copy()
    for col, vals in rows.items():
        out[col] = vals
    out.to_csv(output_csv, index=False)
    return out


def main():
    args = parse_args()
    device, torch_dtype = get_device_and_dtype()

    if not os.path.exists(args.results_csv):
        raise FileNotFoundError(f"Results CSV not found at: {args.results_csv}")

    print(f"Reading evaluation pipeline outputs from: {args.results_csv}")
    results_df = pd.read_csv(args.results_csv)
    if args.answer_column not in results_df.columns:
        raise KeyError(f"Column '{args.answer_column}' not found in results CSV. "
                       f"Available columns: {list(results_df.columns)}")
    if args.max_samples:
        results_df = results_df.head(args.max_samples).reset_index(drop=True)

    print(f"Loading reference dataset: {DATASET_NAME} ({args.split} split)...")
    raw_ds = load_dataset("allenai/sciq", split=args.split)

    print(f"Loading Judge LLM: {args.judge_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.judge_model, torch_dtype=torch_dtype).to(device).eval()

    os.makedirs(args.output_dir, exist_ok=True)
    input_name = os.path.splitext(os.path.basename(args.results_csv))[0]
    output_csv = os.path.join(args.output_dir, f"evaluated_{input_name}.csv")

    rows = {"ground_truth": [], "judge_verdict": [], "judge_reasoning": [], "judge_raw": []}

    print(f"Judging '{args.answer_column}' for {len(results_df)} rows...")
    for n, (_, row) in enumerate(results_df.iterrows(), 1):
        item = raw_ds[int(row["item_idx"])]

        presented_question = _clean(row.get("question", ""))
        original_question = _clean(row.get("original_question", "")) or _clean(item["question"])
        model_answer = str(row[args.answer_column])

        # Guard against judging against the wrong split / mismatched indices.
        if _clean(item["question"]) != original_question:
            raise ValueError(f"Row {n}: original question in CSV does not match the dataset at this index. "
                             f"Check that --split matches the generation run.")

        # Prefer the reference stored in the CSV; fall back to the dataset.
        ref_text = _clean(row.get("correct_answer", ""))
        if not ref_text or ref_text.lower() == "nan":
            ref_text = _clean(item.get("correct_answer", "")) or NO_REFERENCE

        if model_answer.strip() in ("", "<empty>", "nan"):
            score, reasoning, raw = 0, "Empty model answer.", ""
        else:
            prompt = JUDGE_PROMPT_TEMPLATE.format(
                original_question=original_question,
                presented_question=presented_question,
                reference_answer=ref_text,
                model_answer=model_answer,
            )
            score, reasoning, raw = evaluate_correctness_with_judge(prompt, model, tokenizer, device)

        rows["ground_truth"].append(ref_text)
        rows["judge_verdict"].append(np.nan if score is None else score)
        rows["judge_reasoning"].append(reasoning)
        rows["judge_raw"].append(raw)

        shown = "PARSE-FAIL" if score is None else score
        print(f"Progress: {n}/{len(results_df)} | Score: {shown} | Reason: {reasoning[:60]}...", flush=True)

        if n % args.checkpoint_every == 0:
            save_checkpoint(results_df, n, rows, output_csv)

    out = save_checkpoint(results_df, len(results_df), rows, output_csv)

    valid = out.dropna(subset=["judge_verdict"])
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    if "bias_type" in out.columns and len(out):
        print(f"Bias variant: {out['bias_type'].iloc[0]}")
    print(f"Judged: {len(valid)} | Unparseable judge outputs (excluded): {len(out) - len(valid)}")
    if len(valid):
        print(f"Overall Model Accuracy (LLM Judge): {valid['judge_verdict'].mean() * 100:.2f}%")

        # Uncertainty should be HIGHER when the answer is wrong, so "incorrect" is the positive class.
        for col in UNCERTAINTY_COLUMNS:
            if col in valid.columns:
                sub = valid.dropna(subset=[col])
                inc = (sub["judge_verdict"] == 0).astype(int)
                print(f"{col}: AUROC (detecting incorrect) = {auroc(sub[col], inc):.4f} | "
                      f"Pearson r vs. correctness = {sub[col].corr(sub['judge_verdict']):.4f}")
    print("=" * 60)
    print(f"Judged evaluation results saved to: {output_csv}")


if __name__ == "__main__":
    main()