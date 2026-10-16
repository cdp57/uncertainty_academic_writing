# Benchmarking Overconfidence in Information Retrieval for Academic Writings
Information retrieval systems deployed over scientific papers may produce claims that factually misrepresent what the authors report. In particular, models may overstate claims of papers and display overconfidence with respect to these claims, leading to safety concerns in high-risk AI-assisted scientific discovery. To provide a method for confidence calibration in LLMs, we introduce a procedure that benchmarks answer correctness, evidence applicability, overt confidence, and latent confidence for scientific question answering. We likewise extend our benchmark to assess model robustness to confirmation bias. Finally, we suggest exploratory persona-based measures that may serve to mitigate LLM overconfidence concerns. 

## Dependencies
In our project, we are coding with Python 3.10 with PyTorch 2.5 to evaluate model performance and confidence on QA datasets with supporting information provided. Other Python packages include 
- Transformers 5.10 for loading the models
- BitsAndBytes 0.46 for converting Gemma 4 models into 4-bit NF4 quantization
- Huggingface_hub 1.0 for accessing Llama 3.2 and Gemma 4 models
- accerlate 1.0 and safetensors 0.5 supporting model loading in GPU environment
- datasets 3.0 to load the datasets on Hugging Face
- random package for shuffling the choices in SciQ dataset
- re package for parsing the answers
- Pandas 2.0 and NumPy 1.26 for data analysis
- Matplotlib 3.8 for data visualization
- tqdm 4.66 for progress display
- sentencepiece 0.1.99 for tokenization

The dependencies of our project can be installed with
```
pip install -r requirements.txt
```
Requirements vary slighly for the HPC implementation (see requirements_hpc.txt)

## Hardware Requirements
We used Google CoLab to run our experiments. To be specific, we used the NVIDIA T4 GPU offered by Google CoLab for free. It contains 16GB memory. A compatible NVIDIA CUDA environment is required for GPU acceleration. 

For more computationally intensive experiments, we utilize Case Western's
High Performance Computing Cluster (HPC). We plan to use an L40/4090 GPU, given that our models have 1-4B parameters. 

## Datasets
We used [SciQ](https://huggingface.co/datasets/allenai/sciq) and [QASPER](https://huggingface.co/datasets/allenai/qasper) datasets in our project. SciQ is a QA dataset with a short paragraph as supporting information. Each question has with 4 choices, each possible answer is 1-3 words, so we just used accuracy to measure model performance. For QASPER dataset, it is a free-form QA with given an academic article as support. The SciQ dataset has 13.7k questions (~11.7k train/1k validation/1k test) and the QASPER dataset has 5k questions (~2.6k train/1k validation/1.4k test). It is worth noting that for the SciQ dataset, we used its test split with 1000 questions, and for QASPER dataset, we used its validation split, which contains 1005 questions, so we used a similar number of questions from both datasets. They can be loaded with
```
from datasets import load_dataset
dataset = load_dataset("allenai/sciq")
```
and
```
from datasets import load_dataset
dataset = load_dataset("allenai/qasper")
```

## Models
We used Llama 3.2 1B and 3B models, and Gemma 4 E2B and E4B models. We loaded Llama 3.2 models using FP16 precision on GPU and FP32 when CUDA is unavailable, and for Gemma 4 models, we used 4-bit NF4 quantization through BitsAndBytes with FP16 computation to reduce memory consumption. In our Google Colab files, the Llama 3.2 models are loaded with 
```
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    device_map="auto"
)
```

The Gemma 4 models are loaded with
```
from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    padding_side="left"
)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map={"": 0},
    dtype=torch.float16
)
```
The HPC scripts follow similarly with slight modifications for the environment (e.g., integration for CLI arguments).

## Experiments

Our Demo 1 experiments will test a preliminary overconfidence benchmark. We will assess [self-report confidence](https://aclanthology.org/2025.findings-acl.1316/) on SciQ and [semantic entropy](https://arxiv.org/abs/2302.09664) on QASPER.

## HPC Experiment Setup

To setup an environment on the CSDS 447 course space, first clone the repo and enter the repo directory
```
git clone https://github.com/cdp57/uncertainty_academic_writing
```

Then run our setup script to create a virtual environment, set the HF cache, and install packages
```
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

Make sure to input your HF token. Then activate the virtual environment if necessary
```
source /mnt/vstor/courses/csds447/$USER/venv/bin/activate
```

Run a sample experiment with 
```
sbatch scripts/run_experiment_sample.sh
```

Run a full demo experiment with 
```
sbatch scripts/run_experiments.sh
```
