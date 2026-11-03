# Assignment 2: Iranian Accent Classification with a Custom Parity-Net

**Word Embeddings · PyTorch · Custom Neural Architectures · NLP**

## Overview

This assignment explores the classification of Iranian accents and dialects using a custom neural network called **Parity-Net**. The model uses token-level character-length parity to organize word embeddings into two groups, then combines them for six-class classification.

> **Note:** This assignment was completed during a period of widespread internet disruption in Iran.

## Assignment Information

- **Course:** Natural Language Processing
- **Language:** Python
- **Framework:** PyTorch
- **Tokenizer:** `HooshvareLab/bert-base-parsbert-uncased`
- **Dataset:** [Iranian Dialect and Sentiment Dataset](https://github.com/SilkRoadAparsin/Sentiment/tree/main/datasets)

## Approach

The implementation consists of the following steps:

1. **Preprocessing:** Normalized Persian and Arabic character variants, replaced zero-width non-joiners, removed duplicate records, and prepared a text-only dataset.
2. **Dataset Preparation:** Selected six balanced accent classes, with 100 samples per class, and created stratified training, validation, and test splits (80% / 10% / 10%).
3. **Tokenization:** Used the ParsBERT tokenizer to convert sentences into token IDs, with padding and truncation to a maximum length of 30.
4. **Model Architecture:** Implemented Parity-Net with two 64-dimensional embedding layers (`emb_even` and `emb_odd`). Token embeddings are assigned according to token-length parity, combined through mean pooling, and passed to a linear classifier.
5. **Training:** Trained the model using Adam and Cross-Entropy Loss for 20 epochs, with a learning rate of 0.001 and batch size of 8.

## Evaluation Results

<badge color="success">Test Accuracy: 60.00%</badge>

The model correctly classified 36 out of 60 test samples.

| Accent | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Dezfuli | 0.38 | 0.60 | 0.46 |
| Pakistani | 0.90 | 0.90 | 0.90 |
| Semnani | 0.70 | 0.70 | 0.70 |
| Shirazi | 0.38 | 0.30 | 0.33 |
| Southern Kurdish | 1.00 | 0.70 | 0.82 |
| Tonekaboni | 0.44 | 0.40 | 0.42 |
| **Macro Average** | **0.63** | **0.60** | **0.61** |

The model performed best on Pakistani and Southern Kurdish samples, while Shirazi was the most challenging class. The results suggest that shared vocabulary, short sentences, and the loss of word-order information through mean pooling may contribute to classification errors.

## Requirements

- Python 3.11+
- PyTorch
- Hugging Face Transformers
- pandas
- NumPy
- scikit-learn
- Matplotlib and Seaborn

## How to Run

1. Clone or download the project.
2. Install the required dependencies.
3. Download `sentiment_data.csv` from the [dataset repository](https://github.com/SilkRoadAparsin/Sentiment/tree/main/datasets).
4. Place the dataset in the expected location.
5. Open and run the assignment notebook.

## Key Learning Outcomes

- Text preprocessing and exploratory analysis for Persian dialect data.
- Using pretrained tokenizers for text classification.
- Designing custom embedding-based neural architectures in PyTorch.
- Evaluating multiclass classifiers and analyzing their errors.
- Understanding the limitations of simple embedding aggregation.

---

*Completed as part of the Natural Language Processing coursework.*