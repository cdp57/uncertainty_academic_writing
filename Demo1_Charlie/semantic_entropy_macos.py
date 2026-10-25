import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

# Set device to Metal (MPS) for M1 GPU acceleration
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Load Datasets and Models
# ==========================================
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-base"

print("Loading Llama-3.2 model and tokenizer...")
# Ensure you ran `huggingface-cli login` in terminal first
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16
).to(device)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

print("Loading NLI cross-encoder...")
nli_tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_ID)
nli_model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_ID).to(device)

# Load QASPER dataset
qasper_ds = load_dataset("allenai/qasper", split="validation")

# ==========================================
# 2. Helper Functions
# ==========================================

def prepare_qasper_prompt(paper_instance):
    title = paper_instance['title']
    abstract = paper_instance['abstract']
    context = f"Title: {title}\nAbstract: {abstract}"
    
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
            output_scores=True
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


def check_entailment(premise, hypothesis):
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    entailment_prob = probs[0, 2].item()
    return entailment_prob > 0.5


def cluster_semantic_equivalence(responses):
    n = len(responses)
    clusters = []

    for i in range(n):
        assigned = False
        for cluster in clusters:
            representative = responses[cluster[0]]
            if check_entailment(responses[i], representative) and check_entailment(representative, responses[i]):
                cluster.append(i)
                assigned = True
                break
        if not assigned:
            clusters.append([i])
            
    return clusters


def calculate_semantic_entropy(log_probs, clusters):
    probs = np.exp(log_probs)
    probs = probs / np.sum(probs)

    cluster_probs = [np.sum(probs[cluster]) for cluster in clusters]
    cluster_probs = np.array(cluster_probs)
    cluster_probs = cluster_probs[cluster_probs > 0]
    return -np.sum(cluster_probs * np.log(cluster_probs))

# ==========================================
# 3. Execution Example
# ==========================================
if __name__ == "__main__":
    NUM_SAMPLES_PER_PROMPT = 5
    
    sample_paper = qasper_ds[0]
    prompt, question = prepare_qasper_prompt(sample_paper)

    if prompt:
        print(f"\nQuestion: {question}\n")
        
        responses, log_probs = sample_responses_and_logprobs(
            prompt, 
            num_samples=NUM_SAMPLES_PER_PROMPT, 
            temperature=0.7
        )
        
        for i, (resp, lp) in enumerate(zip(responses, log_probs)):
            print(f"Sample [{i+1}]: {resp} (LogProb: {lp:.4f})")
            
        clusters = cluster_semantic_equivalence(responses)
        print(f"\nDiscovered {len(clusters)} Semantic Equivalence Cluster(s): {clusters}")
        
        se_score = calculate_semantic_entropy(log_probs, clusters)
        print(f"\nCalculated Semantic Entropy: {se_score:.4f}")