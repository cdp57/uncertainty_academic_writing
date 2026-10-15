import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Load Datasets and Models
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"  # Used for semantic equivalence check

print(f"Loading {MODEL_ID} model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto"
)

# Set up tokenizer padding configuration
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading NLI cross-encoder...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")

# Load QASPER dataset
qasper_ds = load_dataset("allenai/qasper", split="test")

# Helper Functions
def prepare_qasper_prompt(paper_instance, full_text=True):
    """Formats a paper context and question into a prompt for Llama 3.2 Instruct."""
    title = paper_instance['title']
    abstract = paper_instance['abstract']
    full_text = paper_instance['full_text']
    
    # Simple context truncation using Abstract (or construct full-text snippet)
    if full_text:
        context = f"Title: {title}\nFull Text: {full_text}"
    else:
        context = f"Title: {title}\nAbstract: {abstract}"
    
    # Grab the first available question
    qas = paper_instance['qas']
    question = qas['question'][0] if len(qas['question']) > 0 else None
    
    if not question:
        return None, None

    messages = [
        {"role": "system", "content": "You are a helpful research assistant. Answer the question succinctly based on the context provided."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt, question


def sample_responses_and_logprobs(prompt, num_samples=5, temperature=0.7, max_new_tokens=64):
    """Generates stochastic generations and computes sequence log-probabilities."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    # Repeat prompt inputs for batch generation
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
            output_scores=True
        )

    # Generated sequence tokens (excluding input tokens)
    gen_tokens = outputs.sequences[:, prompt_len:]
    
    # Stack logits for output tokens (shape: [num_samples, gen_len, vocab_size])
    logits = torch.stack(outputs.scores, dim=1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    # Gather log probability for generated token IDs
    target_log_probs = torch.gather(log_probs, 2, gen_tokens.unsqueeze(-1)).squeeze(-1)

    # Compute total normalized sequence log-probabilities
    seq_log_probs = []
    text_responses = []

    for i in range(num_samples):
        # Mask out padding tokens if EOS was reached early
        eos_indices = (gen_tokens[i] == tokenizer.eos_token_id).nonzero()
        length = eos_indices[0].item() if len(eos_indices) > 0 else gen_tokens.shape[1]
        
        valid_log_probs = target_log_probs[i, :length]
        
        # Mean log-prob per token to normalize for generation length
        seq_log_prob = valid_log_probs.sum().item() / max(length, 1)
        seq_log_probs.append(seq_log_prob)

        text = tokenizer.decode(gen_tokens[i, :length], skip_special_tokens=True).strip()
        text_responses.append(text)

    return text_responses, np.array(seq_log_probs)


def check_entailment(premise, hypothesis):
    """Determines if premise entails hypothesis using NLI DeBERTa (Returns True/False)."""
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(nli_model.device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    # NLI model label 2 is usually 'entailment' in deberta-v3-base NLI
    probs = torch.softmax(logits, dim=-1)
    entailment_prob = probs[0, 2].item()
    return entailment_prob > 0.5


def cluster_semantic_equivalence(responses):
    """Groups responses into semantic equivalence classes using Bidirectional Entailment."""
    n = len(responses)
    clusters = [] # list of lists of response indices

    for i in range(n):
        assigned = False
        for cluster in clusters:
            representative = responses[cluster[0]]
            
            # Check bidirectional entailment (A -> B and B -> A)
            if check_entailment(responses[i], representative) and check_entailment(representative, responses[i]):
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            
    return clusters


def calculate_semantic_entropy(log_probs, clusters):
    """Computes semantic entropy over semantic cluster distributions."""
    # Convert sequence log-probs into normalized sequence probabilities
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)

    # Compute probability mass P(c) per cluster
    cluster_probs = []
    for cluster in clusters:
        p_c = np.sum(probs[cluster])
        cluster_probs.append(p_c)

    # Semantic Entropy calculation
    cluster_probs = np.array(cluster_probs)
    cluster_probs = cluster_probs[cluster_probs > 0]  # avoid log(0)
    se = -np.sum(cluster_probs * np.log(cluster_probs))
    
    return se

# Pipeline Example
if __name__ == "__main__":
    NUM_SAMPLES_PER_PROMPT = 5
    NUM_PROMPTS = 5

    for idx in range(NUM_PROMPTS):
        print(f"\n=== Processing Prompt {idx+1}/{NUM_PROMPTS} ===")
        
        # Process sample instance from QASPER
        sample_paper = qasper_ds[idx]
        prompt, question = prepare_qasper_prompt(sample_paper)

        if prompt:
            print(f"\nQuestion: {question}\n")
            
            # Generate multiple stochastic responses with log-probs
            responses, log_probs = sample_responses_and_logprobs(
                prompt, 
                num_samples=NUM_SAMPLES_PER_PROMPT, 
                temperature=0.7
            )
            
            for i, (resp, lp) in enumerate(zip(responses, log_probs)):
                print(f"Sample [{i+1}]: {resp} (LogProb: {lp:.4f})")
                
            # Cluster semantic equivalents using NLI
            clusters = cluster_semantic_equivalence(responses)
            print(f"\nDiscovered {len(clusters)} Semantic Equivalence Cluster(s): {clusters}")
            
            # Compute Semantic Entropy
            se_score = calculate_semantic_entropy(log_probs, clusters)
            print(f"\nCalculated Semantic Entropy: {se_score:.4f}")