import os
import argparse
import torch
import numpy as np
import pandas as pd
from scipy.special import logsumexp, softmax
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

MODEL_MAP = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
}

NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Semantic Entropy on QA Datasets (QASPER / SciQ)")
    parser.add_argument("--model", type=str, default="qwen3-1.7b",
                        help="Key from MODEL_MAP or a custom Hugging Face ID.")
    parser.add_argument("--dataset", type=str, default="qasper", choices=["qasper", "sciq"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-samples", type=int, default=5, help="Sampled outputs per prompt (M)")
    parser.add_argument("--num-runs", type=int, default=1, help="Independent runs per question")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-papers", type=int, default=None, help="Limit items evaluated (None = all)")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--abstract", action="store_false", default=True,
                        help="QASPER: use full paper text as context instead of the abstract.")
    parser.add_argument("--max-context-tokens", type=int, default=4000,
                        help="Truncate QASPER context to this many tokens (guards against OOM).")
    parser.add_argument("--only-answerable", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip unanswerable questions (use --no-only-answerable to include them).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


# ------------------------------------------------------------------------------
# Dataset helpers
# ------------------------------------------------------------------------------
def is_qasper_answerable(paper, q_idx):
    """Answerable if at least one annotation has unanswerable == False."""
    try:
        container = paper["qas"]["answers"][q_idx]
        for ann in container.get("answer", []):
            if not ann.get("unanswerable", False):
                return True
        return False
    except (KeyError, IndexError, TypeError, AttributeError):
        return False


def is_sciq_answerable(item):
    ans = item.get("correct_answer", "")
    return bool(ans and str(ans).strip())


def qasper_full_text(paper):
    """QASPER full_text is a dict: {'section_name': [...], 'paragraphs': [[...], ...]}."""
    ft = paper.get("full_text", None)
    if not ft:
        return ""
    parts = []
    for name, paras in zip(ft.get("section_name", []), ft.get("paragraphs", [])):
        if name:
            parts.append(str(name))
        parts.extend(p for p in paras if p)
    return "\n".join(parts)


def truncate_to_tokens(text, tokenizer, max_tokens):
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_chat_prompt(content, tokenizer, model_id):
    messages = [{"role": "user", "content": content}]
    kwargs = {}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = False  # avoid <think> blocks eating the token budget
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    except Exception:
        return f"{content}\n\nAnswer:"


def prepare_qasper_prompt(paper, q_idx, tokenizer, model_id, full_text, max_context_tokens):
    title = paper.get("title", "")
    questions = paper["qas"]["question"]
    if not questions or q_idx >= len(questions):
        return None, None
    question = questions[q_idx]

    if full_text:
        body = truncate_to_tokens(qasper_full_text(paper), tokenizer, max_context_tokens)
        context = f"Title: {title}\nFull Text: {body}"
    else:
        context = f"Title: {title}\nAbstract: {paper.get('abstract', '')}"

    content = (
        "Answer the following question in a single brief but complete sentence.\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    return build_chat_prompt(content, tokenizer, model_id), question


def prepare_sciq_prompt(item, tokenizer, model_id):
    support = item.get("support", "").strip()
    question = item.get("question", "").strip()
    if support:
        content = ("Answer the following question as briefly as possible.\n"
                   f"Context:\n{support}\n\nQuestion: {question}")
    else:
        content = ("Answer the following question as briefly as possible.\n"
                   f"Question: {question}")
    return build_chat_prompt(content, tokenizer, model_id), question


# ------------------------------------------------------------------------------
# Sampling
# ------------------------------------------------------------------------------
def get_eos_ids(model, tokenizer):
    ids = set()
    gen_eos = model.generation_config.eos_token_id
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif gen_eos:
        ids.update(gen_eos)
    if tokenizer.eos_token_id is not None:
        ids.add(tokenizer.eos_token_id)
    return ids


def sample_responses_and_logprobs(prompt, model, tokenizer, device, eos_ids,
                                  num_samples=5, temperature=0.7, max_new_tokens=64):
    # The chat template already inserts special tokens (e.g. BOS), so don't add them twice.
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids.repeat(num_samples, 1),
            attention_mask=inputs.attention_mask.repeat(num_samples, 1),
            do_sample=True,
            temperature=temperature,
            top_k=0,      # override model-default top-k/top-p so we sample from the full distribution
            top_p=1.0,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=True,  # raw model logits (before temperature / filtering)
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_tokens = outputs.sequences[:, prompt_len:]
    logits = torch.stack(outputs.logits, dim=1).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)

    seq_logps, lengths, texts, token_keys = [], [], [], []
    for i in range(num_samples):
        toks = gen_tokens[i].tolist()
        length = len(toks)
        for pos, t in enumerate(toks):
            if t in eos_ids:
                length = pos + 1  # include the EOS token, exclude everything after (padding)
                break
        seq_logps.append(token_logps[i, :length].sum().item())
        lengths.append(length)
        token_keys.append(tuple(toks[:length]))
        text = tokenizer.decode(toks[:length], skip_special_tokens=True).strip()
        texts.append(text if text else "<empty>")

    return texts, np.array(seq_logps), np.array(lengths), token_keys


def greedy_response(prompt, model, tokenizer, device, eos_ids, max_new_tokens=64):
    """Deterministic (argmax) answer, used as the 'best' answer whose correctness we judge."""
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,  # clear sampling-only defaults to avoid warnings
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_logits=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    toks = outputs.sequences[0, prompt_len:].tolist()
    length = len(toks)
    for pos, t in enumerate(toks):
        if t in eos_ids:
            length = pos + 1
            break

    logits = torch.stack(outputs.logits, dim=1).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, 2, outputs.sequences[:, prompt_len:].unsqueeze(-1)).squeeze(-1)
    seq_logp = token_logps[0, :length].sum().item()

    text = tokenizer.decode(toks[:length], skip_special_tokens=True).strip()
    return (text if text else "<empty>"), seq_logp


# ------------------------------------------------------------------------------
# NLI clustering
# ------------------------------------------------------------------------------
def entailment_index(nli_model):
    for idx, label in nli_model.config.id2label.items():
        if label.lower() == "entailment":
            return int(idx)
    raise ValueError(f"No 'entailment' label in {nli_model.config.id2label}")


@torch.no_grad()
def entails_batch(pairs, nli_model, nli_tokenizer, device, ent_idx):
    """pairs: list of (premise, hypothesis). Returns list[bool] (argmax == entailment)."""
    if not pairs:
        return []
    premises, hypotheses = zip(*pairs)
    inputs = nli_tokenizer(list(premises), list(hypotheses), padding=True, truncation=True,
                           return_tensors="pt").to(device)
    preds = nli_model(**inputs).logits.argmax(dim=-1)
    return (preds == ent_idx).tolist()


def cluster_semantic_equivalence(responses, question, nli_model, nli_tokenizer, device, ent_idx):
    """Greedy clustering with bidirectional entailment; the question is prepended to each answer."""
    def norm(s):
        return s.strip().lower()

    clusters = []
    for i, resp in enumerate(responses):
        # Candidate clusters we can't resolve with string matching
        pairs, cand = [], []
        matched = None
        for c_idx, cluster in enumerate(clusters):
            rep = responses[cluster[0]]
            if norm(resp) == norm(rep):
                matched = c_idx
                break
            if resp == "<empty>" or rep == "<empty>":
                continue  # an empty answer is never equivalent to a non-empty one
            a, b = f"{question} {resp}", f"{question} {rep}"
            pairs.extend([(a, b), (b, a)])
            cand.append(c_idx)

        if matched is None and pairs:
            results = entails_batch(pairs, nli_model, nli_tokenizer, device, ent_idx)
            for k, c_idx in enumerate(cand):
                if results[2 * k] and results[2 * k + 1]:
                    matched = c_idx
                    break

        if matched is None:
            clusters.append([i])
        else:
            clusters[matched].append(i)
    return clusters


# ------------------------------------------------------------------------------
# Uncertainty metrics
# ------------------------------------------------------------------------------
def predictive_entropy(seq_logps):
    """Monte Carlo estimate: -mean log p(s_i)."""
    return float(-np.mean(seq_logps))


def length_normalized_predictive_entropy(seq_logps, lengths):
    """Each sequence's log-prob is divided by its own length before averaging."""
    return float(-np.mean(seq_logps / np.maximum(lengths, 1)))


def semantic_entropy_weighted(seq_logps, token_keys, clusters):
    """
    Probability-weighted SE over unique sampled sequences:
    cluster log-prob = logsumexp of its unique members' sequence log-probs,
    normalised across clusters, then Shannon entropy.
    Duplicate samples are deduplicated so frequency isn't double-counted.
    """
    cluster_logps = []
    for c in clusters:
        seen, lps = set(), []
        for i in c:
            if token_keys[i] not in seen:
                seen.add(token_keys[i])
                lps.append(seq_logps[i])
        cluster_logps.append(logsumexp(lps))
    p = softmax(np.array(cluster_logps))
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def semantic_entropy_discrete(clusters, num_samples):
    """Frequency-based SE: p_c = |c| / M (ignores log-probs)."""
    p = np.array([len(c) for c in clusters], dtype=float) / num_samples
    return float(-np.sum(p * np.log(p)))


# ------------------------------------------------------------------------------
# Per-question evaluation
# ------------------------------------------------------------------------------
def evaluate_question(prompt, question, ctx):
    runs = {"se": [], "se_disc": [], "pe": [], "ln_pe": [], "clusters": []}
    first_responses = None

    # One greedy answer per question (deterministic, so repeating it per run would be redundant)
    greedy_text, greedy_logp = greedy_response(
        prompt, ctx["model"], ctx["tokenizer"], ctx["device"], ctx["eos_ids"],
        max_new_tokens=ctx["max_new_tokens"],
    )

    for run_idx in range(ctx["num_runs"]):
        responses, seq_logps, lengths, token_keys = sample_responses_and_logprobs(
            prompt, ctx["model"], ctx["tokenizer"], ctx["device"], ctx["eos_ids"],
            num_samples=ctx["num_samples"], temperature=ctx["temperature"],
            max_new_tokens=ctx["max_new_tokens"],
        )
        clusters = cluster_semantic_equivalence(
            responses, question, ctx["nli_model"], ctx["nli_tokenizer"], ctx["nli_device"], ctx["ent_idx"]
        )
        runs["se"].append(semantic_entropy_weighted(seq_logps, token_keys, clusters))
        runs["se_disc"].append(semantic_entropy_discrete(clusters, len(responses)))
        runs["pe"].append(predictive_entropy(seq_logps))
        runs["ln_pe"].append(length_normalized_predictive_entropy(seq_logps, lengths))
        runs["clusters"].append(len(clusters))
        if run_idx == 0:
            first_responses = responses

    fmt = lambda xs: " || ".join(f"{v:.6f}" for v in xs)
    return {
        "question": question,
        "greedy_response": greedy_text,
        "greedy_log_prob": greedy_logp,
        "model_response": first_responses[0],
        "all_responses": " || ".join(first_responses),
        "semantic_entropy": float(np.mean(runs["se"])),
        "semantic_entropy_std": float(np.std(runs["se"])) if len(runs["se"]) > 1 else float("nan"),
        "discrete_semantic_entropy": float(np.mean(runs["se_disc"])),
        "predictive_entropy": float(np.mean(runs["pe"])),
        "length_normalized_predictive_entropy": float(np.mean(runs["ln_pe"])),
        "num_clusters": float(np.mean(runs["clusters"])),
        "num_runs": ctx["num_runs"],
        "se_per_run": fmt(runs["se"]),
        "discrete_se_per_run": fmt(runs["se_disc"]),
        "pe_per_run": fmt(runs["pe"]),
        "ln_pe_per_run": fmt(runs["ln_pe"]),
        "clusters_per_run": " || ".join(str(c) for c in runs["clusters"]),
    }


def append_row(row, csv_path):
    pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not os.path.exists(csv_path), index=False)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
        model_id, torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.device_count() > 1 else None,
    )
    if torch.cuda.device_count() <= 1:
        model = model.to(device)
    model.eval()
    eos_ids = get_eos_ids(model, tokenizer)

    # NLI model in float32: DeBERTa in fp16 can be unstable (especially on MPS/CPU)
    print(f"Loading NLI model: {NLI_MODEL_ID}...")
    nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
    nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(device).eval()
    ent_idx = entailment_index(nli_model)
    print(f"NLI label map: {nli_model.config.id2label} (entailment index = {ent_idx})")

    print(f"Loading dataset: {dataset_name} ({args.split} split)...")
    if dataset_name == "qasper":
        dataset = load_dataset("allenai/qasper", revision="refs/convert/parquet", split=args.split)
    else:
        dataset = load_dataset("allenai/sciq", split=args.split)

    total_items = len(dataset) if args.max_papers is None else min(args.max_papers, len(dataset))

    ctx = dict(model=model, tokenizer=tokenizer, device=device, eos_ids=eos_ids,
               nli_model=nli_model, nli_tokenizer=nli_tokenizer, nli_device=device, ent_idx=ent_idx,
               num_samples=args.num_samples, num_runs=args.num_runs,
               temperature=args.temperature, max_new_tokens=args.max_new_tokens)

    os.makedirs(args.output_dir, exist_ok=True)
    clean_model = args.model.replace("/", "_").replace("-", "_")
    csv_path = os.path.join(args.output_dir, f"results_{dataset_name}_{clean_model}.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)  # rows are appended incrementally; start fresh

    se_scores, pe_scores, ln_pe_scores = [], [], []
    skipped = 0
    print(f"Processing {total_items} items from {dataset_name} "
          f"(Only Answerable: {args.only_answerable} | Runs: {args.num_runs})...")

    for idx in range(total_items):
        item = dataset[idx]

        if dataset_name == "qasper":
            work = []
            for q_idx in range(len(item["qas"]["question"])):
                if args.only_answerable and not is_qasper_answerable(item, q_idx):
                    skipped += 1
                    continue
                prompt, question = prepare_qasper_prompt(
                    item, q_idx, tokenizer, model_id, args.abstract, args.max_context_tokens)
                if prompt:
                    work.append((prompt, question, {"paper_idx": idx, "question_idx": q_idx}))
        else:
            if args.only_answerable and not is_sciq_answerable(item):
                skipped += 1
                continue
            prompt, question = prepare_sciq_prompt(item, tokenizer, model_id)
            work = [(prompt, question, {"item_idx": idx, "correct_answer": item.get("correct_answer", "")})]

        for prompt, question, meta in work:
            row = evaluate_question(prompt, question, ctx)
            row = {"dataset": dataset_name, **meta, **row}
            append_row(row, csv_path)  # checkpoint every question
            se_scores.append(row["semantic_entropy"])
            pe_scores.append(row["predictive_entropy"])
            ln_pe_scores.append(row["length_normalized_predictive_entropy"])

        print(f"Progress: {idx + 1}/{total_items} | Evaluated: {len(se_scores)} | Skipped: {skipped}", flush=True)

    print(f"\nCompleted evaluation over {len(se_scores)} questions.")
    if se_scores:
        print(f"Mean Semantic Entropy:                    {np.mean(se_scores):.4f} (Std: {np.std(se_scores):.4f})")
        print(f"Mean Predictive Entropy:                  {np.mean(pe_scores):.4f} (Std: {np.std(pe_scores):.4f})")
        print(f"Mean Length-Normalized Predictive Entropy: {np.mean(ln_pe_scores):.4f} (Std: {np.std(ln_pe_scores):.4f})")
        print(f"Results saved to CSV: {csv_path}")
    else:
        print("No questions were evaluated. If this is QASPER, check the answerability schema "
              "(print(paper['qas']['answers'][0])).")


if __name__ == "__main__":
    main()