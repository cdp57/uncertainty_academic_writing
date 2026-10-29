# CSDS 447 — Demo 2: Charlie's Contribution

- **Entropy Measures**: The included scripts compute predictive entropy, length normalized predictive entropy, and semantic entropy. See the evaluate_semantic_entropy.py script. 
- **LLM-as-a-Judge**: The first sampled answer from the entropy calculation procedure is judged by Llama-3.2-3b, which is provided with both the output answer and the correct answer. See LLM_judge.py.
- **Plotting**: The plotting script incorporates the Rouge-L metric for accuracy and AUROC and AURC as Responsible AI metrics. See plot_metrics_comparison.py.
- **Confirmation Bias**: An LLM, Llama-3.2-3b, is prompted to generate biased questions in the biased_questions.py file. These questions either suggest the model respond with the correct answer or an incorrect answer for the SciQ dataset. The rest of the pipeline runs accordingly.

## Scripts

The environment can be setup on the HPC in the same manner as Demo1. The scripts folders provides a set of scripts needed to run the experiments. Also experiments can be run on a single L40 or 4090 GPU, although high-performance nodes are not requested for basic computational functions (e.g., plotting).

## AI Use Disclosure

An AI model, CWRU Gemini, was used for basic coding assistance (e.g., assistance parsing CLI inputs) and debugging (e.g., understanding why predictive entropy was lower than semantic entropy when this is theoretically impossible)

An example prompt would be: "Review this code and identify if there are any methodological errors in the computation of predictive and semantic entropy?"

Any AI-generated or modified code was manually review for correctness.
