# CSDS 447 Demo 2 — Results Summary

Generated 2026-10-25 14:45 EDT | Results: /home/cxg438/447_demo2/results

Models: Llama 3.2 1B/3B and Qwen3 1.7B/4B. Datasets: SciQ test and QASPER validation. All questions include supporting context.

## 1. Vanilla baseline — reference metrics

This is the straightforward baseline. Values are proportions from 0 to 1, aggregated across the five generated answers per question.

| Model | Dataset | Accuracy (exact match) | Token F1 |
| --- | --- | --- | --- |
| Llama 1B | QASPER | 0.026 | 0.087 |
| Llama 1B | SciQ | 0.298 | 0.368 |
| Llama 3B | QASPER | 0.134 | 0.359 |
| Llama 3B | SciQ | 0.578 | 0.685 |
| Qwen 1.7B | QASPER | 0.059 | 0.182 |
| Qwen 1.7B | SciQ | 0.448 | 0.594 |
| Qwen 4B | QASPER | 0.192 | 0.418 |
| Qwen 4B | SciQ | 0.706 | 0.811 |

## 2. How the four baselines compare

Higher is better for the two reference-based scores below. Standard = one-prompt answer + confidence; Structured = separate answer and confidence; P(True) = brainstormed answers with a separate truth-probability step; Caution = a prompt warning against overconfidence.

### Exact-match accuracy

| Model | Dataset | Standard | Structured | P(True) | Caution |
| --- | --- | --- | --- | --- | --- |
| Llama 1B | QASPER | 0.026 | 0.051 | 0.062 | 0.004 |
| Llama 1B | SciQ | 0.298 | 0.233 | 0.195 | 0.176 |
| Llama 3B | QASPER | 0.134 | 0.142 | 0.134 | 0.124 |
| Llama 3B | SciQ | 0.578 | 0.635 | 0.622 | 0.565 |
| Qwen 1.7B | QASPER | 0.059 | 0.003 | 0.011 | 0.073 |
| Qwen 1.7B | SciQ | 0.448 | 0.109 | 0.074 | 0.424 |
| Qwen 4B | QASPER | 0.192 | 0.155 | 0.139 | 0.191 |
| Qwen 4B | SciQ | 0.706 | 0.682 | 0.678 | 0.709 |

### Answer token F1

| Model | Dataset | Standard | Structured | P(True) | Caution |
| --- | --- | --- | --- | --- | --- |
| Llama 1B | QASPER | 0.087 | 0.184 | 0.206 | 0.064 |
| Llama 1B | SciQ | 0.368 | 0.316 | 0.268 | 0.224 |
| Llama 3B | QASPER | 0.359 | 0.367 | 0.360 | 0.352 |
| Llama 3B | SciQ | 0.685 | 0.735 | 0.725 | 0.671 |
| Qwen 1.7B | QASPER | 0.182 | 0.153 | 0.150 | 0.237 |
| Qwen 1.7B | SciQ | 0.594 | 0.214 | 0.167 | 0.577 |
| Qwen 4B | QASPER | 0.418 | 0.402 | 0.391 | 0.418 |
| Qwen 4B | SciQ | 0.811 | 0.793 | 0.791 | 0.811 |

### Best-performing baseline in each configuration

| Model | Dataset | Best exact-match baseline | Accuracy | Gain vs standard |
| --- | --- | --- | --- | --- |
| Llama 1B | QASPER | P(True) | 0.062 | +3.6 pp |
| Llama 1B | SciQ | Standard | 0.298 | +0.0 pp |
| Llama 3B | QASPER | Structured | 0.142 | +0.8 pp |
| Llama 3B | SciQ | Structured | 0.635 | +5.7 pp |
| Qwen 1.7B | QASPER | Caution | 0.073 | +1.4 pp |
| Qwen 1.7B | SciQ | Standard | 0.448 | +0.0 pp |
| Qwen 4B | QASPER | Standard | 0.192 | +0.0 pp |
| Qwen 4B | SciQ | Caution | 0.709 | +0.3 pp |

## 3. Pre-evaluation: what the models reported

### Mean reported confidence (1–10)

These numbers measure how confident the models said they were. High confidence does not automatically mean high accuracy.

| Model | Dataset | Standard | Structured | Caution |
| --- | --- | --- | --- | --- |
| Llama 1B | QASPER | 6.03 | 7.19 | 5.35 |
| Llama 1B | SciQ | 7.00 | 6.31 | 5.32 |
| Llama 3B | QASPER | 8.51 | 7.80 | 8.17 |
| Llama 3B | SciQ | 9.55 | 8.46 | 9.28 |
| Qwen 1.7B | QASPER | 7.83 | 7.53 | 7.58 |
| Qwen 1.7B | SciQ | 9.64 | 8.99 | 9.35 |
| Qwen 4B | QASPER | 7.95 | 8.35 | 7.62 |
| Qwen 4B | SciQ | 9.26 | 7.33 | 9.07 |

### P(True): average assigned probability

P(True) is the model's estimated probability for option A (True) when evaluating its first brainstormed answer. It is NOT measured correctness.

| Model | Dataset | Mean P(True) | Available questions |
| --- | --- | --- | --- |
| Llama 1B | QASPER | 0.189 | 100.0% |
| Llama 1B | SciQ | 0.061 | 100.0% |
| Llama 3B | QASPER | 0.727 | 100.0% |
| Llama 3B | SciQ | 0.654 | 100.0% |
| Qwen 1.7B | QASPER | 0.929 | 100.0% |
| Qwen 1.7B | SciQ | 0.948 | 100.0% |
| Qwen 4B | QASPER | 0.885 | 100.0% |
| Qwen 4B | SciQ | 0.984 | 100.0% |

### Answer consistency across five responses

Shows the standard baseline only. 'Unanimous' means all five normalized answers were identical, not necessarily correct.

| Model | Dataset | Different answers / 5 | Unanimous | Answer parse rate |
| --- | --- | --- | --- | --- |
| Llama 1B | QASPER | 3.72 | 4.4% | 100.0% |
| Llama 1B | SciQ | 2.84 | 22.6% | 100.0% |
| Llama 3B | QASPER | 3.45 | 14.1% | 100.0% |
| Llama 3B | SciQ | 1.87 | 53.6% | 99.9% |
| Qwen 1.7B | QASPER | 1.79 | 56.8% | 100.0% |
| Qwen 1.7B | SciQ | 1.21 | 84.1% | 99.9% |
| Qwen 4B | QASPER | 1.72 | 63.9% | 100.0% |
| Qwen 4B | SciQ | 1.09 | 93.8% | 99.9% |

## 4. LLM-as-a-Judge: does the model approve its own answers?

Each model judges its own answers. The table uses the standard baseline so it is easy to compare with Section 1.

| Model | Dataset | Exact match | Judge says TRUE | Difference | Valid judge outputs | Judge confidence |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 1B | QASPER | 2.6% | 47.9% | +45.2 pp | 27.4% | 8.41 |
| Llama 1B | SciQ | 29.8% | 53.0% | +23.2 pp | 77.8% | 6.99 |
| Llama 3B | QASPER | 13.4% | 66.7% | +53.3 pp | 85.2% | 7.65 |
| Llama 3B | SciQ | 57.8% | 85.6% | +27.9 pp | 99.7% | 8.79 |
| Qwen 1.7B | QASPER | 5.9% | 1.9% | -4.1 pp | 98.5% | 3.11 |
| Qwen 1.7B | SciQ | 44.8% | 13.9% | -30.9 pp | 97.0% | 3.14 |
| Qwen 4B | QASPER | 19.2% | 78.8% | +59.6 pp | 100.0% | 7.37 |
| Qwen 4B | SciQ | 70.6% | 94.5% | +23.8 pp | 100.0% | 8.03 |

A positive difference means the model labeled more answers TRUE than exact matching counted as correct. It does not by itself show that the model judged incorrectly; correct paraphrases may fail exact matching. Judge TRUE rates exclude responses whose verdict could not be parsed.

## 5. Main observations — talking points for the demo

- SciQ: Qwen 4B had the highest standard-baseline exact-match accuracy (70.6%) and token F1 of 81.1%.
- QASPER: Qwen 4B had the highest standard-baseline exact-match accuracy (19.2%) and token F1 of 41.8%.
- QASPER was harder under the exact-match measure: every model scored lower than on SciQ (average drop 40.5 pp). QASPER's longer, more varied answers also make exact matching stricter.
- Llama model size: the larger model's standard-baseline accuracy changed by +28.0 pp on SciQ; changed by +10.8 pp on QASPER.
- Qwen model size: the larger model's standard-baseline accuracy changed by +25.8 pp on SciQ; changed by +13.2 pp on QASPER.
- Structured vs standard: higher exact-match accuracy in 3/8 model–dataset comparisons. It is not a universal improvement.
- P(True) vs standard: higher exact-match accuracy in 2/8 model–dataset comparisons. It is not a universal improvement.
- Caution vs standard: higher exact-match accuracy in 2/8 model–dataset comparisons. It is not a universal improvement.
- Self-judging can be lenient: on QASPER, Qwen 4B labeled 78.8% of its valid standard-baseline judgments TRUE, versus 19.2% reference exact match. This gap needs semantic/human review before calling it judge error.
- Judge format reliability matters: Llama 1B on QASPER produced valid standard-baseline verdicts for only 27.4% of the evaluated answers.
- Reported confidence is not accuracy: on QASPER, Llama 3B reported 8.51/10 average standard-baseline confidence while exact-match accuracy was 13.4%.
- P(True) scores also need calibration: the largest mean assigned probability was 0.984 for Qwen 4B on SciQ. That score is a model estimate, not a measured accuracy rate.
- Sampling consistency is visible even without checking correctness: Qwen 4B on SciQ gave the same normalized standard-baseline answer in all five responses for 93.8% of eligible questions.

## 6. How to interpret the report

- Accuracy (exact match): the generated answer must match a reference after normalizing the text. A correct synonym or paraphrase may be counted wrong.
- Token F1: how much the answer's words overlap with the reference. It is more flexible than exact match, but it cannot fully judge scientific meaning.
- Self-judge TRUE rate: how often a model says its own answer is correct. This is NOT independent accuracy; valid judge-output coverage matters.
- Confidence and P(True): what the model reports or assigns, not verified probabilities of being correct. Compare them with reference scores carefully.
- Each baseline produced five answers per question. The tables aggregate those answers; they do not represent five separate experiments over the dataset.
- QASPER token F1 measures answer overlap only (not official evidence F1). Long-form answers make wording-based comparisons especially limited.
- All generation and judging used supporting context; the report and result tables do not include that support text.
