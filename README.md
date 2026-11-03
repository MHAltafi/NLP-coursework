# Natural Language Processing — University of Tehran

This repository contains my coursework and implementations for the **Natural Language Processing** course at **University of Tehran (Semester 4042)**.

The assignments cover a range of NLP concepts, from statistical language modeling and neural network architectures to Transformer interpretability, text generation, and Retrieval-Augmented Generation (RAG). Each assignment includes a Jupyter Notebook with the corresponding implementation and experiments.

Each assignment folder also contains a dedicated `README.md` with a more detailed description of its implementation, experiments, results, and key findings.

> **Note:** The first three assignments were completed during a period of widespread internet disruption in Iran.

## Table of Contents

- [Assignments](#assignments)
  - [Assignment 1 — Character-Level Language Modeling](#assignment-1--character-level-language-modeling)
  - [Assignment 2 — Iranian Accent Classification with Parity-Net](#assignment-2--iranian-accent-classification-with-parity-net)
  - [Assignment 3 — Autoregressive Text Generation and Multi-Head Causal Attention](#assignment-3--autoregressive-text-generation-and-multi-head-causal-attention)
  - [Assignment 4.1 — LLM Interpretability with BERT](#assignment-41--llm-interpretability-with-bert)
  - [Assignment 4.2 — RAG and Prompting Strategy Comparison](#assignment-42--rag-and-prompting-strategy-comparison)
- [Repository Structure](#repository-structure)
- [Technologies](#technologies)
- [Topics Covered](#topics-covered)

## Assignments

### Assignment 1 — Character-Level Language Modeling

The first assignment focuses on **statistical language modeling** and character-level n-gram models for a word-decoding task.

The implementation includes:

- Text preprocessing and vocabulary preparation.
- A random decoder baseline.
- Unigram and length-aware unigram language models.
- Bigram language modeling.
- A hybrid n-gram model combining unigram, bigram, trigram, and higher-order n-grams.
- Laplace smoothing and hierarchical backoff.
- Bidirectional context and word-length-aware probability estimation.
- Evaluation through average incorrect guesses.

**Notebook:** [NLP_CA1.ipynb](https://github.com/MHAltafi/NLP-coursework/blob/main/Assignment1/NLP_CA1.ipynb)

### Assignment 2 — Iranian Accent Classification with Parity-Net

The second assignment focuses on **text classification and custom neural network architectures**, using word embeddings to classify sentences into six Iranian accent and dialect classes.

The implementation includes:

- Dataset exploration, preprocessing, and Persian text normalization.
- Class distribution analysis and text statistics.
- Tokenization using ParsBERT's tokenizer.
- Custom PyTorch Dataset and DataLoader implementations.
- A custom neural architecture called **Parity-Net**, using separate even- and odd-token embedding tables.
- Model training and evaluation.
- Classification metrics and confusion matrix analysis.

**Notebook:** [NLP_CA2.ipynb](https://github.com/MHAltafi/NLP-coursework/blob/main/Assignment2/NLP_CA2.ipynb)

### Assignment 3 — Autoregressive Text Generation and Multi-Head Causal Attention

The third assignment explores **autoregressive text generation** and the internal mechanics of Transformer attention.

The implementation includes:

- Numerically stable softmax and cross-entropy calculations.
- Greedy decoding and random sampling.
- Temperature-based probability scaling.
- Autoregressive generation with end-of-sequence handling.
- Perplexity evaluation and entropy analysis.
- Query, Key, and Value projections.
- Scaled dot-product attention.
- Causal masking to prevent access to future tokens.
- Multi-head causal self-attention implemented from scratch.

**Notebook:** [NLP_CA3.ipynb](https://github.com/MHAltafi/NLP-coursework/blob/main/Assignment3/NLP_CA3.ipynb)

### Assignment 4.1 — LLM Interpretability with BERT

The first part of Assignment 4 investigates **how sentiment information is represented across BERT's hidden layers** and how fine-tuning affects internal representations.

The implementation includes:

- Extracting `[CLS]` representations from all BERT layers.
- PCA and UMAP visualization of hidden representations.
- Layer-wise linear probing using logistic regression.
- Fine-tuning BERT for binary sentiment classification on SST-2.
- Comparing pretrained and fine-tuned representations.
- Inter-layer cosine similarity analysis.
- Investigating the evolution of sentiment-related information across Transformer layers.

**Notebook:** [NLP_CA4_Q1.ipynb](https://github.com/MHAltafi/NLP-coursework/blob/main/Assignment4-1/NLP_CA4_Q1.ipynb)

### Assignment 4.2 — RAG and Prompting Strategy Comparison

The second part of Assignment 4 focuses on **Retrieval-Augmented Generation (RAG)** and evaluates how different prompting strategies affect question-answering performance.

The implementation includes:

- Preparing a SQuAD-based document corpus and evaluation dataset.
- Generating document embeddings using BGE.
- Building a vector database with LanceDB.
- Semantic retrieval of relevant context.
- Implementing No-RAG, Zero-shot RAG, Few-shot RAG, and Chain-of-Thought RAG approaches.
- Comparing generated answers using ROUGE metrics.
- Evaluating semantic consistency using a Natural Language Inference (NLI) model.
- LoRA fine-tuning of Qwen and evaluating the fine-tuned model against RAG approaches.

**Notebook:** [NLP_CA4_Q2.ipynb](https://github.com/MHAltafi/NLP-coursework/blob/main/Assignment4-2/NLP_CA4_Q2.ipynb)

## Repository Structure

```text
NLP-coursework/
│
├── Assignment1/
│   ├── NLP_CA1.ipynb
│   ├── README.md
│   └── hw1_word_bank.txt
│
├── Assignment2/
│   ├── NLP_CA2.ipynb
│   └── README.md
│
├── Assignment3/
│   ├── NLP_CA3.ipynb
│   └── README.md
│
├── Assignment4-1/
│   ├── figs/
│   ├── .gitignore
│   ├── NLP_CA4_Q1.ipynb
│   └── README.md
│
├── Assignment4-2/
│   ├── .gitignore
│   ├── NLP_CA4_Q2.ipynb
│   └── README.md
│
└── README.md
```

## Technologies

The coursework uses a range of Python tools and machine learning frameworks, including:

- **Python**
- **Jupyter Notebook**
- **NumPy**
- **Pandas**
- **Matplotlib and Seaborn**
- **Scikit-learn**
- **PyTorch**
- **Hugging Face Transformers**
- **Hugging Face Datasets**
- **UMAP-learn**
- **LanceDB**
- **Sentence-Transformers**
- **PEFT (LoRA)**
- **ROUGE and NLI**

## Topics Covered

| Assignment | Area | Topics |
|---|---|---|
| Assignment 1 | Statistical Language Modeling | Unigram, Bigram, N-gram Models, Smoothing, Backoff |
| Assignment 1 | Text Preprocessing | Normalization, Vocabulary Construction, Train/Test Split |
| Assignment 2 | Text Classification | Accent and Dialect Classification |
| Assignment 2 | Neural Networks | Embeddings, Custom Architectures, Model Training |
| Assignment 2 | Model Evaluation | Accuracy, Precision, Recall, F1-score |
| Assignment 3 | Language Generation | Autoregressive Generation, Greedy Decoding, Sampling, Temperature, Perplexity |
| Assignment 3 | Transformer Architecture | Self-Attention, Multi-Head Attention, Causal Masking |
| Assignment 4.1 | Model Interpretability | Hidden Representations, PCA, UMAP, Linear Probing |
| Assignment 4.1 | Transfer Learning | BERT Fine-Tuning, Layer-wise Representation Analysis |
| Assignment 4.2 | Retrieval-Augmented Generation | Vector Search, Embeddings, Context Retrieval |
| Assignment 4.2 | Prompt Engineering | Zero-shot, Few-shot, Chain-of-Thought |
| Assignment 4.2 | Model Evaluation | ROUGE, Natural Language Inference (NLI) |
| Assignment 4.2 | Parameter-Efficient Fine-Tuning | LoRA, Qwen Fine-Tuning |

---

*Coursework completed as part of the Natural Language Processing course at the University of Tehran.*