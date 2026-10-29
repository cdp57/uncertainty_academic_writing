import os
import argparse
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Supported model mapping
MODEL_MAP = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Entropy on QA Datasets (QASPER / SciQ)")
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
        default="qasper",
        choices=["qasper", "sciq"],
        help="Select dataset to evaluate on: 'qasper' or 'sciq'."
    )
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (e.g., 'validation', 'test')")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sampled outputs per prompt (M)")
    parser.add_argument("--num-runs", type=int, default=1, help="Number of independent evaluation runs per question for variance control")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max output length per generation")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers/items to evaluate (None for all)")
    parser.add_argument("--output-dir", type=str, default="results", help="Folder to save CSV results")
    parser.add_argument("--no-context", action="store_true", help="For QASPER: Use full text context instead of abstract")
    parser.add_argument("--only-answerable", action="store_true", default=True, help="Skip unanswerable questions (default: True)")
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def is_qasper_answerable(paper_instance, question_idx):
    """
    Determines if a QASPER question is answerable.
    In QASPER, each question has answer annotations from practitioners.
    A question is deemed answerable if at least one annotation has 'unanswerable' set to False.
    """
    try:
        answers_container = paper_instance['qas']['answers'][question_idx]
        answer_annotations = answers_container.get('answer', [])
        for ann in answer_annotations:
            if not ann.get('unanswerable', False):
                return True
        return False
    except (KeyError, IndexError, TypeError):
        return False


def is_sciq_answerable(item):
    """Determines if a SciQ item has a non-empty correct answer."""
    correct_ans = item.get('correct_answer', '')
    return bool(correct_ans and str(correct_ans).strip())


# ==============================================================================
# [UPDATED]: Preamble Suppression for QASPER
# ==============================================================================
def prepare_qasper_prompt(paper_instance, question_idx, tokenizer, full_text=True):
    title = paper_instance.get('title', '')
    abstract = paper_instance.get('abstract', '')

    if full_text:
        full_text_val = paper_instance.get('full_text', '')
        context = f"Title: {title}\nFull Text: {full_text_val}"
    else:
        context = f"Title: {title}\nAbstract: {abstract}"
    
    questions = paper_instance['qas']['question']
    if not questions or question_idx >= len(questions):
        return None, None

    question = questions[question_idx]

    content = (
        "Answer the research question concisely using only information from the provided context. "
        "Do NOT include introductory filler, conversational preambles (e.g., 'Based on the paper...'), "
        "or explanations of your reasoning. State only the direct answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    messages = [{"role": "user", "content": content}]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{content}\n\nAnswer:"
    
    return prompt, question


# ==============================================================================
# [UPDATED]: Preamble Suppression & Short Direct Format for SciQ
# ==============================================================================
def prepare_sciq_prompt(instance, tokenizer):
    support = instance.get('support', '').strip()
    question = instance.get('question', '').strip()

    if support:
        content = (
            "Answer the science question directly in as few words as possible (1-5 words). "
            "Do NOT include introductory phrases, explanation, or context repetition.\n\n"
            f"Context:\n{support}\n\nQuestion: {question}"
        )
    else:
        content = (
            "Answer the science question directly in as few words as possible (1-5 words). "
            "Do NOT include introductory phrases, explanation, or context repetition.\n\n"
            f"Question: {question}"
        )

    messages = [{"role": "user", "content": content}]

    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"{content}\n\nAnswer:"

    return prompt, question


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
        
        # Include <eos> token in valid slice length
        if len(eos_indices) > 0:
            length = eos_indices[0].item() + 1
        else:
            length = gen_tokens.shape[1]
        
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
    """Checks NLI entailment with fast-path string matching."""
    # Fast path for identical strings
    if premise.strip().lower() == hypothesis.strip().lower():
        return True

    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    
    # Label 1 = entailment for cross-encoder/nli-deberta-v3-base
    return probs[0, 1].item() > 0.5


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
    """Computes standard Predictive Entropy over sequence log-probabilities."""
    probs = np.exp(unnorm_log_probs - np.max(unnorm_log_probs))
    probs = probs / np.sum(probs)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def calculate_semantic_entropy(unnorm_log_probs, clusters):
    """
    Computes Semantic Entropy by summing probabilities of sequences within the same cluster.
    Uses unnormalized sequence log-probabilities P(s) to guarantee SE <= PE.
    """
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

    print(f"Using compute device: {device}")
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

    print(f"Loading NLI model: {NLI_MODEL_ID}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(
        NLI_MODEL_ID,
        torch_dtype=torch_dtype
    ).to(device)

    print(f"Loading dataset: {dataset_name} ({args.split} split)...")
    if dataset_name == "qasper":
        dataset = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=args.split)
    elif dataset_name == "sciq":
        dataset = load_dataset("sciq", split=args.split)

    total_items = len(dataset) if args.max_papers is None else min(args.max_papers, len(dataset))
    
    se_scores = []
    pe_scores = []
    ln_pe_scores = []
    results_data = []

    print(f"Processing {total_items} items from {dataset_name} (Only Answerable: {args.only_answerable} | Runs: {args.num_runs})...")

    if dataset_name == "qasper":
        skipped_unanswerable = 0
        for paper_idx in range(total_items):
            paper = dataset[paper_idx]
            questions = paper['qas']['question']
            
            for q_idx in range(len(questions)):
                # Filter out unanswerable QASPER questions
                if args.only_answerable and not is_qasper_answerable(paper, q_idx):
                    skipped_unanswerable += 1
                    continue

                prompt, question = prepare_qasper_prompt(paper, q_idx, tokenizer, full_text=args.no_context)
                if not prompt:
                    continue

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

                se_scores.append(mean_se)
                pe_scores.append(mean_pe)
                ln_pe_scores.append(mean_ln_pe)

                results_data.append({
                    "dataset": dataset_name,
                    "paper_idx": paper_idx,
                    "question_idx": q_idx,
                    "question": question,
                    "model_response": first_run_responses[0],
                    "all_responses": " || ".join(first_run_responses),
                    "semantic_entropy": mean_se,
                    "semantic_entropy_std": std_se,
                    "predictive_entropy": mean_pe,
                    "length_normalized_predictive_entropy": mean_ln_pe,
                    "num_clusters": float(np.mean(clusters_runs)),
                    "num_runs": args.num_runs,
                    "se_per_run": " || ".join([f"{v:.6f}" for v in se_runs]),
                    "pe_per_run": " || ".join([f"{v:.6f}" for v in pe_runs]),
                    "ln_pe_per_run": " || ".join([f"{v:.6f}" for v in ln_pe_runs]),
                    "clusters_per_run": " || ".join([str(c) for c in clusters_runs]),
                })

            print(f"Progress: {paper_idx + 1}/{total_items} papers processed | Evaluated: {len(se_scores)} | Skipped Unanswerable: {skipped_unanswerable}", flush=True)

    elif dataset_name == "sciq":
        skipped_unanswerable = 0
        for item_idx in range(total_items):
            item = dataset[item_idx]

            # Filter out invalid SciQ items
            if args.only_answerable and not is_sciq_answerable(item):
                skipped_unanswerable += 1
                continue

            prompt, question = prepare_sciq_prompt(item, tokenizer)

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

            se_scores.append(mean_se)
            pe_scores.append(mean_pe)
            ln_pe_scores.append(mean_ln_pe)

            results_data.append({
                "dataset": dataset_name,
                "item_idx": item_idx,
                "question": question,
                "correct_answer": item.get('correct_answer', ''),
                "model_response": first_run_responses[0],
                "all_responses": " || ".join(first_run_responses),
                "semantic_entropy": mean_se,
                "semantic_entropy_std": std_se,
                "predictive_entropy": mean_pe,
                "length_normalized_predictive_entropy": mean_ln_pe,
                "num_clusters": float(np.mean(clusters_runs)),
                "num_runs": args.num_runs,
                "se_per_run": " || ".join([f"{v:.6f}" for v in se_runs]),
                "pe_per_run": " || ".join([f"{v:.6f}" for v in pe_runs]),
                "ln_pe_per_run": " || ".join([f"{v:.6f}" for v in ln_pe_runs]),
                "clusters_per_run": " || ".join([str(c) for c in clusters_runs]),
            })

            print(f"Progress: {item_idx + 1}/{total_items} questions processed | Evaluated: {len(se_scores)} | Skipped Invalid: {skipped_unanswerable}", flush=True)

    print(f"\nCompleted evaluation over {len(se_scores)} answerable questions.")
    
    if se_scores:
        print(f"Mean Semantic Entropy:                     {np.mean(se_scores):.4f} (Std: {np.std(se_scores):.4f})")
        print(f"Mean Predictive Entropy:                    {np.mean(pe_scores):.4f} (Std: {np.std(pe_scores):.4f})")
        print(f"Mean Length-Normalized Predictive Entropy:  {np.mean(ln_pe_scores):.4f} (Std: {np.std(ln_pe_scores):.4f})")
        
        os.makedirs(args.output_dir, exist_ok=True)
        clean_model = args.model.replace("/", "_").replace("-", "_")
        csv_path = os.path.join(args.output_dir, f"results_{dataset_name}_{clean_model}.csv")
        df = pd.DataFrame(results_data)
        df.to_csv(csv_path, index=False)
        print(f"Results saved to CSV: {csv_path}")


if __name__ == "__main__":
    main()