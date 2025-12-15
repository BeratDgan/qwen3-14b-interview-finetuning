# AI Technical Interview Coach (Fine-Tuned Qwen 3 14B)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Unsloth-green)
![Model](https://img.shields.io/badge/Model-Qwen%203%2014B-purple)
![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Model%20Link-yellow)](https://huggingface.co/beratdgan/Qwen3-14B-Interview-Coach)

##  Project Overview

This project provides a specialized Large Language Model (LLM) fine-tuned to act as a **Senior Technical Interviewer** and **Evaluator**. Built upon the **Qwen 3 14B** architecture, this model is designed to define the "Gold Standard" for high-level engineering interviews similar to those at top-tier tech companies.

**Core Functionality:**
Unlike generic chat models, this model is specifically trained to generate **Evaluation Rubrics**. It takes an interview question as input and outputs the *ideal answer key*, covering technical depth, Big O notation, and behavioral (STAR) structures. This output is then used by the backend system to grade candidate responses.

##  Repository Structure

```bash
├── Interview-Prep-Qwen3_14B.ipynb   # The complete training, testing, and export notebook (Colab/Unsloth)
└── README.md                   # Project documentation
````

##  Data Curation & Training Stats

The model was fine-tuned using **Unsloth** on a Tesla A100 GPU. The dataset curation was a critical part of this project's success:

  * **Raw Source:** Over **50,000** raw interview questions were scraped and collected from various open technical platforms.
  * **Curation Process:** Through a rigorous data cleaning, deduplication, and synthetic enhancement pipeline, this massive corpus was distilled down to **10,708 high-quality instruction-response pairs**.
  * **Result:** A focused dataset ensuring the model learns only from clear, solvable, and relevant questions.

| Metric | Value |
| :--- | :--- |
| **Final Dataset Size** | 10,708 Pairs |
| **Trainable Parameters** | 256,901,120 (1.71%) |
| **Final Training Loss** | **0.5128** (High Convergence) |
| **Optimization** | Unsloth (2x Faster Training, 60% Less VRAM) |

##  Installation & Usage

### Prerequisites

  * **Python 3.10+**
  * **Ollama** (for local inference) OR **llama-cpp-python**

### Option 1: Local Inference with Ollama (Recommended)

1.  **Download Model:** Get the `.gguf` file from the [Hugging Face Repository](https://huggingface.co/beratdgan/Qwen3-14B-Interview-Coach).
2.  **Create Modelfile:**
    ```dockerfile
    FROM ./qwen3-14b.Q4_K_M.gguf
    SYSTEM "You are a Senior Technical Interviewer. Your task is to provide the 'Gold Standard' answer key and evaluation criteria for the given interview question."
    PARAMETER temperature 0.2
    PARAMETER num_ctx 4096
    ```
3.  **Run:**
    ```bash
    ollama create interview-coach -f Modelfile
    ollama run interview-coach
    ```

### Option 2: Python Integration (Backend)

Use this script to integrate the model into your FastAPI/Django backend.

```bash
pip install llama-cpp-python
```

```python
from llama_cpp import Llama

# 1. Load the Model (Offload to GPU if available)
llm = Llama(
    model_path="./qwen3-14b.Q4_K_M.gguf",
    n_ctx=4096,
    n_gpu_layers=-1, 
    verbose=False
)

# 2. Dynamic Prompt Construction
# These variables would come from your Database/RAG system
company_name = "Some Tech Company "
difficulty_level = "Hard"
question = "How would you design a rate limiter?"

prompt = f"""<|im_start|>system
You are a Senior Technical Interviewer at {company_name}. Your task is to define the 'Gold Standard' answer for a {difficulty_level}-level question.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""

# 3. Generate Rubric
output = llm(prompt, max_tokens=512, stop=["<|im_end|>"], temperature=0.2)
print(output['choices'][0]['text'])
```

##  License

  * **Source Code (Notebooks/Scripts):** Licensed under **Apache 2.0**. You are free to modify and use the code.
  * **Model Weights (GGUF):** The fine-tuned model weights are released under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial).
      *  **Allowed:** Research, personal projects, educational use.
      *  **Prohibited:** Commercial use without explicit permission.

##  Acknowledgments

  * **Unsloth AI:** For the optimized training pipeline.
  * **Qwen Team:** For the robust base model architecture.

-----

*Project maintained by Berat Doğan.*

```