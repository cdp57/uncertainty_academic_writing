import os
import argparse
import re
import json
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

BIAS_PROMPT_TEMPLATE = """You are an expert in cognitive bias and survey design.
Your task is to rewrite a question into two leading variants that exhibit confirmation bias.

Original Question: {question}
Correct Answer: {correct_answer}
Incorrect Answer: {incorrect_answer}

Instructions:
1. Write "biased_correct": Rewrite the question so it strongly leads/nudges the respondent toward assuming "{correct_answer}" is correct.
2. Write "biased_incorrect": Rewrite the question so it strongly leads/nudges the respondent toward assuming "{incorrect_answer}" is correct.

Return ONLY a valid JSON object with no markdown surrounding it:
{{"biased_correct": "<biased question leaning toward correct answer>", "biased_incorrect": "<biased question leaning toward incorrect answer>"}}"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Biased Question Variants for QASPER and SciQ")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Hugging Face model ID")
    parser.add_argument("--output-dir", type=str, default="data", help="Directory to save CSV outputs")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples to process per dataset (None for all)")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Max output tokens for generation")
    return parser.parse_args()


def get_device_and_dtype():
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    else:
        return torch.device("cpu"), torch.float32


def extract_qasper_references(q_answers_list):
    refs = []
    for ans_item in q_answers_list:
        ans = ans_item.get('answer', {})
        if ans.get('unanswerable', False):
            refs.append("Unanswerable")
        elif ans.get('yes_no') is not None:
            refs.append("Yes" if ans.get('yes_no') else "No")
        elif ans.get('free_response'):
            refs.append(ans['free_response'])
        elif ans.get('extractive_spans'):
            spans = [s.strip() for s in ans['extractive_spans'] if s.strip()]
            if spans:
                refs.append(", ".join(spans))
    return " | ".join(refs) if refs else "No answer"


def parse_json_response(raw_text):
    json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            if "biased_correct" in data and "biased_incorrect" in data:
                return data["biased_correct"].strip(), data["biased_incorrect"].strip()
        except Exception:
            pass
    return raw_text.strip(), raw_text.strip()


def generate_bias_pair(prompt, model, tokenizer, device, max_new_tokens=128):
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        formatted_prompt = f"{prompt}\n\nResponse:"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic framing
            pad_token_id=tokenizer.pad_token_id
        )

    gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return parse_json_response(response_text)


def process_sciq(dataset, model, tokenizer, device, max_samples=None, max_new_tokens=128):
    print("\n--- Processing SciQ Test Set ---")
    results = []
    items_to_run = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for idx in range(items_to_run):
        item = dataset[idx]
        question = item['question']
        correct_ans = item['correct_answer']
        incorrect_ans = item.get('distractor1', 'an incorrect alternative answer')

        prompt = BIAS_PROMPT_TEMPLATE.format(
            question=question,
            correct_answer=correct_ans,
            incorrect_answer=incorrect_ans
        )

        b_correct, b_incorrect = generate_bias_pair(prompt, model, tokenizer, device, max_new_tokens)

        results.append({
            "dataset": "sciq",
            "item_idx": idx,
            "original_question": question,
            "correct_answer": correct_ans,
            "incorrect_answer": incorrect_ans,
            "biased_correct_question": b_correct,
            "biased_incorrect_question": b_incorrect
        })

        print(f"SciQ [{idx + 1}/{items_to_run}] Processed", flush=True)

    return pd.DataFrame(results)


def process_qasper(dataset, model, tokenizer, device, max_samples=None, max_new_tokens=128):
    print("\n--- Processing QASPER Validation Set ---")
    results = []
    papers_to_run = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    processed_count = 0

    for paper_idx in range(papers_to_run):
        paper = dataset[paper_idx]
        questions = paper['qas']['question']
        answers = paper['qas']['answers']

        for q_idx, question in enumerate(questions):
            correct_ans = extract_qasper_references(answers[q_idx]['answer'])
            incorrect_ans = f"an opposing or incorrect claim to: {correct_ans[:50]}"

            prompt = BIAS_PROMPT_TEMPLATE.format(
                question=question,
                correct_answer=correct_ans,
                incorrect_answer=incorrect_ans
            )

            b_correct, b_incorrect = generate_bias_pair(prompt, model, tokenizer, device, max_new_tokens)

            results.append({
                "dataset": "qasper",
                "paper_idx": paper_idx,
                "question_idx": q_idx,
                "original_question": question,
                "correct_answer": correct_ans,
                "incorrect_answer": incorrect_ans,
                "biased_correct_question": b_correct,
                "biased_incorrect_question": b_incorrect
            })

            processed_count += 1

        print(f"QASPER Paper [{paper_idx + 1}/{papers_to_run}] | Total Questions: {processed_count}", flush=True)

    return pd.DataFrame(results)


def main():
    args = parse_args()
    device, torch_dtype = get_device_and_dtype()

    print(f"Using compute device: {device}")
    print(f"Loading Llama Model: {args.model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch_dtype
    ).to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. SciQ Test Set
    sciq_ds = load_dataset("sciq", split="test")
    sciq_df = process_sciq(sciq_ds, model, tokenizer, device, args.max_samples, args.max_new_tokens)
    sciq_out = os.path.join(args.output_dir, "sciq_biased_questions.csv")
    sciq_df.to_csv(sciq_out, index=False)
    print(f"Saved SciQ biased questions to: {sciq_out}")

    # 2. QASPER Validation Set
    qasper_ds = load_dataset("allenai/qasper", revision="refs/convert/parquet", split="validation")
    qasper_df = process_qasper(qasper_ds, model, tokenizer, device, args.max_samples, args.max_new_tokens)
    qasper_out = os.path.join(args.output_dir, "qasper_biased_questions.csv")
    qasper_df.to_csv(qasper_out, index=False)
    print(f"Saved QASPER biased questions to: {qasper_out}")


if __name__ == "__main__":
    main()