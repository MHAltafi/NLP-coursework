# Assignment 4 – Question 2: RAG and Prompting Strategy Comparison

## Overview

This assignment implements a **Retrieval-Augmented Generation (RAG)** system and compares different prompting strategies for question answering.

Using a subset of the **SQuAD** dataset, the experiment evaluates No-RAG, Zero-shot RAG, Few-shot RAG, and Chain-of-Thought (CoT) RAG. It also compares these approaches against a Qwen model fine-tuned using **LoRA**.

Generated answers are evaluated using both lexical and semantic metrics: ROUGE and RoBERTa-based Natural Language Inference (NLI).

## Assignment Information

| Item | Description |
|---|---|
| Dataset | SQuAD |
| Embedding model | `BAAI/bge-small-en-v1.5` |
| Generation model | `Qwen/Qwen3.5-2B` |
| NLI model | `FacebookAI/roberta-large-mnli` |
| Vector database | LanceDB |
| Fine-tuning method | LoRA |

## Dataset

The experiment uses four CSV files derived from SQuAD:

| File | Samples | Purpose |
|---|---:|---|
| `squad_corpus_100.csv` | 100 | Retrieval corpus |
| `squad_gold_20.csv` | 20 | Evaluation questions and reference answers |
| `finetune_train.csv` | 4,561 | Fine-tuning dataset |
| `finetune_val.csv` | 303 | Fine-tuning validation dataset |

## Implemented Approaches

### 1. Vector Database and Semantic Retrieval

- Defined a LanceDB schema containing passage text, vector embeddings, and document IDs.
- Generated passage embeddings using `BAAI/bge-small-en-v1.5`.
- Stored the 100 Wikipedia passages in LanceDB.
- Implemented a semantic search function to retrieve the top 5 relevant passages for each question.

### 2. Answer Generation Strategies

Four approaches were implemented and evaluated:

| Method | Description |
|---|---|
| **No RAG** | Answers questions using the model's internal knowledge without retrieved context. |
| **Zero-shot RAG** | Uses the top 5 retrieved passages and a prompt requesting short, direct answers. |
| **Few-shot RAG** | Combines retrieved passages with three example question-answer pairs. |
| **CoT RAG** | Uses retrieved passages and prompts the model to reason step by step before providing a final answer. |

Predictions, reference answers, and retrieved contexts were saved as CSV files.

### 3. ROUGE Evaluation

Evaluated generated answers using:

- ROUGE-1
- ROUGE-2
- ROUGE-L

The average scores across the 20 evaluation questions were calculated for each method.

### 4. RoBERTa NLI Evaluation

Used `FacebookAI/roberta-large-mnli` to classify each generated answer against its reference answer into:

- Entailment
- Neutral
- Contradiction

The percentage of each category was calculated for all four approaches.

### 5. LoRA Fine-Tuning

Fine-tuned `Qwen/Qwen3.5-2B` using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

| Parameter | Value |
|---|---|
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `v_proj` |
| Epochs | 1 |
| Learning rate | 2e-4 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Precision | FP16 when supported |
| Gradient checkpointing | Enabled |

The fine-tuned model was evaluated without retrieval using the same ROUGE and NLI metrics.

## Evaluation Results

### ROUGE Scores

<text color="secondary" size="xs">Average scores over 20 evaluation questions.</text>

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | Average |
|---|---:|---:|---:|---:|
| **Zero-shot RAG** | **0.6513** | **0.4768** | **0.6513** | **0.5931** |
| CoT RAG | 0.6369 | 0.4074 | 0.6369 | 0.5604 |
| Few-shot RAG | 0.5233 | 0.4278 | 0.5233 | 0.4915 |
| No RAG | 0.1279 | 0.0143 | 0.1252 | 0.0891 |
| Fine-tuned Model | 0.1279 | 0.0143 | 0.1252 | 0.0891 |

### RoBERTa NLI Results

Percentage of predictions assigned to each NLI category.

| Method | Entailment | Neutral | Contradiction |
|---|---:|---:|---:|
| **CoT RAG** | **65%** | 25% | 10% |
| Zero-shot RAG | 60% | 40% | **0%** |
| Few-shot RAG | 45% | 40% | 15% |
| No RAG | 10% | 40% | 50% |
| Fine-tuned Model | 10% | 40% | 50% |

## Key Findings

- **Zero-shot RAG** achieves the highest ROUGE scores, likely because its concise prompt focuses the model on answering directly from the retrieved context.
- **CoT RAG** achieves the highest entailment rate (65%), indicating stronger semantic agreement with reference answers according to the NLI evaluator.
- RAG substantially improves performance compared with the No-RAG baseline. ROUGE-1 increases from 0.1279 to 0.6513 with Zero-shot RAG.
- The contradiction rate decreases from 50% without RAG to 0% with Zero-shot RAG.
- ROUGE and NLI identify different strengths: Zero-shot performs best in lexical overlap, while CoT achieves the highest semantic entailment.
- The fine-tuned model does not outperform the RAG approaches in this experiment. The relatively small fine-tuning dataset provides limited factual knowledge compared with retrieving relevant external passages at inference time.

**Conclusion:** Zero-shot RAG is the best overall choice when lexical matching, concise answers, and a low contradiction rate are priorities. CoT RAG is useful when semantic agreement is more important than exact wording. Fine-tuning is better suited to adapting model behavior, style, or task-specific patterns than providing new factual knowledge.

## Output Files

The notebook generates the following CSV files:

| File | Contents |
|---|---|
| `no_rag_results.csv` | Baseline predictions |
| `zero_shot_rag_results.csv` | Zero-shot predictions and retrieved contexts |
| `few_shot_rag_results.csv` | Few-shot predictions and retrieved contexts |
| `cot_rag_results.csv` | CoT reasoning, final answers, and retrieved contexts |
| `rouge_comparison.csv` | ROUGE comparison |
| `final_comparison_rouge_nli.csv` | Combined ROUGE and NLI results |
| `finetuned_model_results.csv` | Fine-tuned model predictions |
| `complete_evaluation_with_finetuned.csv` | Final comparison including the fine-tuned model |

## Requirements

Main libraries:

- `torch`
- `transformers`
- `datasets`
- `numpy`
- `pandas`
- `matplotlib`
- `lancedb`
- `sentence-transformers`
- `rouge-score`
- `peft`
- `accelerate`
- `bitsandbytes`
- `torchao`

A CUDA-enabled GPU is recommended for model inference and LoRA fine-tuning.

## How to Run

1. Open `NLP_CA4_Q2.ipynb` in Google Colab.
2. Install the required dependencies.
3. Mount Google Drive and place the four dataset CSV files in the directory expected by the notebook.
4. Run the notebook cells sequentially to build the vector database, retrieve passages, generate answers, and evaluate the prompting strategies.
5. Run the LoRA fine-tuning section to train and save the adapter.
6. Complete the final evaluation to compare the fine-tuned model against the RAG approaches.

**Note:** The current notebook uses Google Colab and Google Drive paths. The dataset files must be available at those paths, and the required Hugging Face models must be accessible or cached.

## Key Learning Outcomes

- Building a semantic retrieval system using embeddings and LanceDB.
- Understanding the role of external context in reducing unsupported answers.
- Comparing Zero-shot, Few-shot, and Chain-of-Thought prompting.
- Evaluating generated answers using lexical and semantic metrics.
- Implementing parameter-efficient fine-tuning with LoRA.
- Understanding the trade-offs between retrieval-based generation and fine-tuning.

---

*This assignment was completed as part of the Natural Language Processing coursework.*