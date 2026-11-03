# Assignment 3: Autoregressive Text Generation and Multi-Head Causal Attention

**Transformer Architecture · Autoregressive Decoding · Attention Mechanisms · NLP**

## Overview

This assignment focuses on implementing two fundamental components of Transformer-based language models from scratch. The first part covers autoregressive text generation and perplexity evaluation, while the second implements a complete multi-head causal self-attention mechanism using Python.

> **Note:** This assignment was completed during a period of widespread internet disruption in Iran.

## Assignment Information

- **Course:** Natural Language Processing
- **Language:** Python 3.11
- **Environment:** Jupyter Notebook
- **External Dataset:** Not required

## Implemented Approaches

### Part 1: Autoregressive Text Generation

Implemented the mathematical and algorithmic components of an autoregressive generation pipeline:

- Numerically stable softmax and cross-entropy loss.
- Greedy decoding and probability-weighted random sampling.
- Temperature-scaled softmax for controlling generation diversity.
- Autoregressive generation loop with EOS token termination.
- Corpus-level perplexity calculation using teacher-forced predictions.
- Entropy analysis across different temperature settings.

**Key Results:**

| Evaluation | Result |
|---|---:|
| Greedy generation | `all` |
| Sampling frequency for highest-probability token | 60.37% |
| Overall corpus perplexity | **2.9018** |

Temperature experiments also demonstrated the expected behavior: lower temperatures concentrate probability on the most likely token, while higher temperatures produce flatter distributions and greater entropy.

### Part 2: Multi-Head Causal Attention

Built a modular multi-head attention layer using basic Python matrix operations, implementing:

- Query, Key, and Value linear projections.
- Splitting projected representations into multiple attention heads.
- Scaled dot-product attention.
- Upper-triangular causal masking to prevent access to future tokens.
- Stable row-wise softmax normalization.
- Head concatenation and final output projection.

**Validation Results:**

- Successfully projected input matrices.
- Correctly split embeddings into two attention heads.
- Verified causal masking on attention score matrices.
- Passed the single-head attention verification.
- Successfully executed the complete multi-head attention pipeline.

The final validation used a four-token sequence with an embedding dimension of 4 and two attention heads.

## Requirements

- Python 3.11+
- Jupyter Notebook or JupyterLab

The implementations use Python's standard libraries, including `math`, `random`, and `json`, without requiring external machine-learning frameworks.

## How to Run

1. Clone or download the project.
2. Open the Assignment 3 notebook in Jupyter Notebook or JupyterLab.
3. Run the cells sequentially to execute the implementations and validation pipelines.

## Key Learning Outcomes

- Understanding the mathematical foundations of autoregressive language generation.
- Implementing decoding strategies and temperature-based sampling.
- Evaluating language-model predictions using cross-entropy and perplexity.
- Understanding multi-head attention and causal masking.
- Building Transformer attention components from scratch using matrix operations.

---

*Completed as part of the Natural Language Processing coursework.*