import os
import argparse
import re
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Supported model mapping
MODEL_MAP = {
    "qwen3_1.7b": "Qwen/Qwen3-1.7B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "llama3.2_1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2_3b": "meta-llama/Llama-3.2-3B-Instruct",
}

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Entropy on Biased QA Datasets")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3-1.7b",
        choices=list(MODEL_MAP.keys()) + list(MODEL_MAP.values()),
        help="Select model option or pass a custom Hugging Face ID."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="sciq",
        choices=["qasper", "sciq"],
        help="Select dataset to evaluate on: 'qasper' or 'sciq'."
    )
    parser.add_argument(
        "--bias-type",
        type=str,
        default="biased_incorrect",
        choices=["biased_correct", "biased_incorrect", "unbiased"],
        help="Bias variant of the question to present to the model."
    )
    parser.add_argument(
        "--biased-input-csv",
        type=str,
        default=None,
        help="Path to biased CSV file (defaults to '{dataset}_biased_questions.csv')."
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (e.g., 'validation', 'test')")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sampled outputs per prompt (M)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent evaluation runs per question")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max output length per generation")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of items to evaluate (None for all)")
    parser.add_argument("--output-dir", type=str, default="results_biased", help="Folder to save CSV results")
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def parse_biased_question_string(val, bias_type):
    """
    Robustly parses question strings from biased CSV fields.
    Handles plain text, complete JSON, and truncated JSON strings.
    """
    if pd.isna(val):
        return ""
    val_str = str(val).strip()

    # 1. Regex match for JSON key (handles complete or truncated JSON)
    pattern = rf'"{bias_type}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, val_str)
    if match:
        return match.group(1).strip()

    # 2. Try JSON loads fallback
    if val_str.startswith("{"):
        try:
            fixed_str = val_str if val_str.endswith("}") else val_str + '"}'
            data = json.loads(fixed_str)
            if bias_type in data:
                return data[bias_type].strip()
        except Exception:
            pass

    # 3. Plain text fallback
    return val_str


def get_question_by_bias_type(row, bias_type):
    """Extracts target question text from row based on bias_type."""
    if bias_type == "unbiased":
        return str(row.get("original_question", row.get("question", ""))).strip()
    elif bias_type == "biased_correct":
        col_val = row.get("biased_correct_question", row.get("original_question", ""))
        return parse_biased_question_string(col_val, "biased_correct")
    elif bias_type == "biased_incorrect":
        col_val = row.get("biased_incorrect_question", row.get("original_question", ""))
        return parse_biased_question_string(col_val, "biased_incorrect")
    return ""


def prepare_qasper_prompt(paper_instance, question_text, tokenizer, full_text=True):
    title = paper_instance.get('title', '')
    abstract = paper_instance.get('abstract', '')

    if full_text:
        full_text_val = paper_instance.get('full_text', '')
        context = f"Title: {title}\nFull Text: {full_text_val}"
    else:
        context = f"Title: {title}\nAbstract: {abstract}"

    content = (
        "Answer the research question concisely using only information from the provided context. "
        "Do NOT include introductory filler, conversational preambles (e.g., 'Based on the paper...'), "
        "or explanations of your reasoning. State only the direct answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {question_text}"
    )

    messages = [{"role": "user", "content": content}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{content}\n\nAnswer:"

    return prompt


def prepare_sciq_prompt(instance, question_text, tokenizer):
    support = instance.get('support', '').strip()

    if support:
        content = (
            "Answer the science question directly in as few words as possible (1-5 words)."
            "Do NOT include introductory phrases, explanation, or context repetition.\n\n"
            "The response should either be \"Yes, the answer is {answer}\" or \"No, the answer is {answer}\"\n\n"
            f"Context:\n{support}\n\nQuestion: {question_text}"
        )
    else:
        content = (
            "Answer the science question directly in as few words as possible (1-5 words). "
            "Do NOT include introductory phrases, explanation, or context repetition.\n\n"
            "The response should either be \"Yes, the answer is {answer}\" or \"No, the answer is {answer}\"\n\n"
            f"Question: {question_text}"
        )

    messages = [{"role": "user", "content": content}]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{content}\n\nAnswer:"

    return prompt


def sample_responses_and_logprobs(prompt, model, tokenizer, device, num_samples=5, temperature=0.7, max_new_tokens=64):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]

    input_ids = inputs.input_ids.repeat(num_samples, 1)
    attention_mask = inputs.attention_mask.repeat(num_samples, 1)

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id
        )

    gen_tokens = outputs.sequences[:, prompt_len:]
    logits = torch.stack(outputs.scores, dim=1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_log_probs = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)

    unnorm_seq_log_probs = []
    gen_lengths = []
    text_responses = []

    for i in range(num_samples):
        eos_indices = (gen_tokens[i] == tokenizer.eos_token_id).nonzero()
        length = eos_indices[0].item() + 1 if len(eos_indices) > 0 else gen_tokens.shape[1]

        valid_log_probs = target_log_probs[i, :length]
        unnorm_log_prob = valid_log_probs.sum().item()

        unnorm_seq_log_probs.append(unnorm_log_prob)
        gen_lengths.append(length)

        text = tokenizer.decode(gen_tokens[i, :length], skip_special_tokens=True).strip()
        if not text:
            text = "<empty>"
        text_responses.append(text)

    return text_responses, np.array(unnorm_seq_log_probs), np.array(gen_lengths)


def check_entailment(premise, hypothesis, nli_model, nli_tokenizer, device):
    if premise.strip().lower() == hypothesis.strip().lower():
        return True

    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0, 1].item() > 0.5  # Label 1 = entailment


def cluster_semantic_equivalence(responses, nli_model, nli_tokenizer, device):
    n = len(responses)
    clusters = []

    for i in range(n):
        assigned = False
        for cluster in clusters:
            representative = responses[cluster[0]]
            if (check_entailment(responses[i], representative, nli_model, nli_tokenizer, device) and
                check_entailment(representative, responses[i], nli_model, nli_tokenizer, device)):
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])

    return clusters


def calculate_predictive_entropy(unnorm_log_probs):
    probs = np.exp(unnorm_log_probs - np.max(unnorm_log_probs))
    probs = probs / np.sum(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def calculate_semantic_entropy(unnorm_log_probs, clusters):
    probs = np.exp(unnorm_log_probs - np.max(unnorm_log_probs))
    probs = probs / np.sum(probs)

    cluster_probs = np.array([np.sum(probs[c]) for c in clusters])
    cluster_probs = cluster_probs[cluster_probs > 0]
    return -np.sum(cluster_probs * np.log(cluster_probs))


def main():
    args = parse_args()
    model_id = MODEL_MAP.get(args.model.lower(), args.model)
    dataset_name = args.dataset.lower()
    device, torch_dtype = get_device_and_dtype()

    # Determine input biased CSV path
    input_csv = args.biased_input_csv if args.biased_input_csv else f"{dataset_name}_biased_questions.csv"
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Biased input file not found at: {input_csv}")

    print(f"Reading biased questions from: {input_csv}")
    biased_df = pd.read_csv(input_csv)

    print(f"Loading reference HF dataset: {dataset_name} ({args.split} split)...")
    if dataset_name == "qasper":
        raw_ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=args.split)
    elif dataset_name == "sciq":
        raw_ds = load_dataset("sciq", split=args.split)

    print(f"Loading generator model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.device_count() > 1 else None
    )
    if torch.cuda.device_count() == 1:
        model = model.to(device)

    print(f"Loading NLI judge model: {NLI_MODEL_ID}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_ID,
        torch_dtype=torch_dtype
    ).to(device)

    total_items = len(biased_df) if args.max_papers is None else min(args.max_papers, len(biased_df))
    results_data = []

    print(f"Evaluating {total_items} items under bias variant: '{args.bias_type}'...")

    for idx in range(total_items):
        row = biased_df.iloc[idx]
        question_text = get_question_by_bias_type(row, args.bias_type)

        if not question_text:
            continue

        # Fetch context from raw HF dataset using item_idx/paper_idx
        if dataset_name == "qasper":
            paper_idx = int(row.get("paper_idx", row.get("item_idx", idx)))
            paper = raw_ds[paper_idx]
            prompt = prepare_qasper_prompt(paper, question_text, tokenizer)
        else: # sciq
            item_idx = int(row.get("item_idx", idx))
            item = raw_ds[item_idx]
            prompt = prepare_sciq_prompt(item, question_text, tokenizer)

        se_runs, pe_runs, ln_pe_runs, clusters_runs = [], [], [], []
        first_run_responses = None

        for run_idx in range(args.num_runs):
            responses, unnorm_log_probs, gen_lengths = sample_responses_and_logprobs(
                prompt, model, tokenizer, device,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens
            )

            clusters = cluster_semantic_equivalence(responses, nli_model, nli_tokenizer, device)

            pe_score = calculate_predictive_entropy(unnorm_log_probs)
            se_score = calculate_semantic_entropy(unnorm_log_probs, clusters)
            mean_len = float(np.mean(gen_lengths))
            ln_pe_score = pe_score / max(mean_len, 1.0)

            se_runs.append(se_score)
            pe_runs.append(pe_score)
            ln_pe_runs.append(ln_pe_score)
            clusters_runs.append(len(clusters))

            if run_idx == 0:
                first_run_responses = responses

        mean_se = float(np.mean(se_runs))
        std_se = float(np.std(se_runs)) if len(se_runs) > 1 else 0.0
        mean_pe = float(np.mean(pe_runs))
        mean_ln_pe = float(np.mean(ln_pe_runs))

        res_dict = {
            "dataset": dataset_name,
            "bias_type": args.bias_type,
            "item_idx": row.get("item_idx", idx),
            "question": question_text,
            "original_question": row.get("original_question", question_text),
            "correct_answer": row.get("correct_answer", ""),
            "incorrect_answer": row.get("incorrect_answer", ""),
            "model_response": first_run_responses[0],
            "all_responses": " || ".join(first_run_responses),
            "semantic_entropy": mean_se,
            "semantic_entropy_std": std_se,
            "predictive_entropy": mean_pe,
            "length_normalized_predictive_entropy": mean_ln_pe,
            "num_clusters": float(np.mean(clusters_runs)),
            "num_runs": args.num_runs,
        }
        
        if "paper_idx" in row:
            res_dict["paper_idx"] = row["paper_idx"]
        if "question_idx" in row:
            res_dict["question_idx"] = row["question_idx"]

        results_data.append(res_dict)

        print(f"Progress: {idx + 1}/{total_items} | Bias: {args.bias_type} | SE: {mean_se:.4f} | PE: {mean_pe:.4f}", flush=True)

    # Save output using standard nomenclature into target output directory
    os.makedirs(args.output_dir, exist_ok=True)
    clean_model = args.model.replace("/", "_").replace("-", "_")
    output_filename = f"results_{dataset_name}_{clean_model}.csv"
    output_path = os.path.join(args.output_dir, output_filename)

    out_df = pd.DataFrame(results_data)
    out_df.to_csv(output_path, index=False)
    print(f"\nSuccessfully saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()