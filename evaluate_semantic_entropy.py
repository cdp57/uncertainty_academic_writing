import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Supported model mapping
MODEL_MAP = {
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "gemma-e2b": "google/gemma-2-2b-it",
    "gemma-e4b": "google/gemma-2-9b-it",
}

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Entropy on QASPER Validation Set")
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2-1b",
        choices=list(MODEL_MAP.keys()) + list(MODEL_MAP.values()),
        help="Select model option (llama3.2-1b, llama3.2-3b, gemma-e2b, gemma-e4b) or pass a custom Hugging Face ID."
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of sampled outputs per prompt (M)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Max output length per generation")
    parser.add_argument("--max-papers", type=int, default=None, help="Limit number of papers to evaluate (None for all)")
    parser.add_argument("--output-dir", type=str, default="figures", help="Folder to save generated histogram figures")
    parser.add_argument("--no-context", action="store_true", help="Use full text context instead of abstract")
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def prepare_qasper_prompt(paper_instance, question_idx, tokenizer, full_text=True):
    title = paper_instance.get('title', '')
    abstract = paper_instance.get('abstract', '')

    if full_text:
        full_text = paper_instance.get('full_text', '')
        context = f"Title: {title}\nFull Text: {full_text}"
    else:
        context = f"Title: {title}\nAbstract: {abstract}"
    
    questions = paper_instance['qas']['question']
    if not questions or question_idx >= len(questions):
        return None, None

    question = questions[question_idx]

    messages = [
        {
            "role": "user",
            "content": f"You are a helpful research assistant. Answer the question succinctly based on the context provided.\n\nContext:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    
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

    seq_log_probs = []
    text_responses = []

    for i in range(num_samples):
        eos_indices = (gen_tokens[i] == tokenizer.eos_token_id).nonzero()
        length = eos_indices[0].item() if len(eos_indices) > 0 else gen_tokens.shape[1]
        
        valid_log_probs = target_log_probs[i, :length]
        seq_log_prob = valid_log_probs.sum().item() / max(length, 1)
        seq_log_probs.append(seq_log_prob)

        text = tokenizer.decode(gen_tokens[i, :length], skip_special_tokens=True).strip()
        text_responses.append(text)

    return text_responses, np.array(seq_log_probs)


def check_entailment(premise, hypothesis, nli_model, nli_tokenizer, device):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return probs[0, 2].item() > 0.5  # Label 2 corresponds to entailment


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


def calculate_semantic_entropy(log_probs, clusters):
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)

    cluster_probs = np.array([np.sum(probs[c]) for c in clusters])
    cluster_probs = cluster_probs[cluster_probs > 0]
    return -np.sum(cluster_probs * np.log(cluster_probs))


def plot_and_save_histogram(se_values, model_alias, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    plt.hist(se_values, bins=15, color='#4C72B0', edgecolor='black', alpha=0.8)
    plt.title(f"Semantic Entropy Distribution on QASPER Validation Set\nModel: {model_alias}", fontsize=13)
    plt.xlabel("Semantic Entropy", fontsize=11)
    plt.ylabel("Paper Question Count", fontsize=11)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    clean_name = model_alias.replace("/", "_").replace("-", "_")
    output_filename = os.path.join(output_dir, f"semantic_entropy_{clean_name}.png")
    
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    plt.close()
    print(f"\nHistogram saved to: {output_filename}")


def main():
    args = parse_args()
    model_id = MODEL_MAP.get(args.model.lower(), args.model)
    device, torch_dtype = get_device_and_dtype()

    print(f"Using compute device: {device}")
    print(f"Loading generator model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype
    ).to(device)

    print(f"Loading NLI model: {NLI_MODEL_ID}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(device)

    print("Loading QASPER validation split...")
    qasper_ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split="validation")
    
    papers_to_process = len(qasper_ds) if args.max_papers is None else min(args.max_papers, len(qasper_ds))
    se_scores = []

    print(f"Processing {papers_to_process} paper(s)...")
    for paper_idx in range(papers_to_process):
        paper = qasper_ds[paper_idx]
        questions = paper['qas']['question']
        
        for q_idx in range(len(questions)):
            prompt, question = prepare_qasper_prompt(paper, q_idx, tokenizer, full_text=args.no_context)
            if not prompt:
                continue

            responses, log_probs = sample_responses_and_logprobs(
                prompt, model, tokenizer, device,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens
            )

            clusters = cluster_semantic_equivalence(responses, nli_model, nli_tokenizer, device)
            se_score = calculate_semantic_entropy(log_probs, clusters)
            se_scores.append(se_score)

        print(f"Progress: {paper_idx + 1}/{papers_to_process} papers processed | Accumulated Questions: {len(se_scores)}", end="\r")

    print(f"\nCompleted evaluation over {len(se_scores)} total questions.")
    
    if se_scores:
        print(f"Mean Semantic Entropy: {np.mean(se_scores):.4f}")
        print(f"Std Semantic Entropy:  {np.std(se_scores):.4f}")
        plot_and_save_histogram(se_scores, args.model, args.output_dir)


if __name__ == "__main__":
    main()