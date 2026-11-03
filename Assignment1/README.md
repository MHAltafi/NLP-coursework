# Assignment 1: Character-Level Language Modeling for Signal Decoding

**Natural Language Processing (NLP) · N-gram Language Models · Character-Level Prediction**

## Overview

This assignment focuses on implementing character-level language models to solve a word-guessing problem called **Signal Decoding**.

> **Note:** This assignment was completed during a period of widespread internet disruption in Iran.

In this scenario, an agent attempts to recover a hidden English word. Initially, only the length of the word is known, and all character positions are masked. At each step, the agent proposes one character:

- If the character exists in the hidden word, all its occurrences are revealed.
- If the guess is incorrect, the agent receives an error.
- The game ends when the word is fully recovered or the maximum number of allowed errors is reached.

The main objective is to develop increasingly sophisticated language models that minimize the average number of incorrect guesses on previously unseen words.

## Implemented Approaches

The assignment explores several decoding strategies, progressing from simple statistical methods to a more advanced language model.

### Question 1: Dataset Preparation

The provided word bank (`hw1_word_bank.txt`) is used to construct the training and evaluation datasets.

Preprocessing includes:

- Converting words to lowercase.
- Keeping only alphabetic words.
- Removing duplicate words.
- Randomly shuffling the vocabulary using `random.seed(1)`.
- Splitting the vocabulary into:
  - **Training set:** All remaining words (approximately 40,000).
  - **Evaluation set:** Exactly 1,000 words.

The two datasets are disjoint, ensuring that evaluation is performed on previously unseen words.

### Question 2: Random Decoder

A baseline strategy that randomly selects an English alphabet character from `a` to `z`, excluding previously guessed characters.

This approach provides a reference point for evaluating the performance of more sophisticated language models.

### Question 3: Character-Level Unigram Model

Calculates the frequency of each character across all training words and selects the most frequent character that has not already been guessed.

This model uses global character statistics without considering character positions or surrounding context.

### Question 4: Length-Aware Unigram Model

Extends the unigram approach by learning separate character frequency distributions for words of different lengths.

If an unseen word length is encountered, the model falls back to the general unigram distribution.

### Question 5: Character-Level Bigram Model

Uses the character immediately preceding an unknown position as context to estimate the probability of the next character.

A special start-of-word token (`$`) is used to handle the first character. When no useful left context is available, the model falls back to unigram frequencies.

### Question 6: Advanced Hybrid N-gram Model

The final approach combines multiple language models to improve prediction accuracy:

- Unigram
- Bigram
- Trigram
- 4-gram
- 5-gram

The model incorporates several techniques:

- **Laplace smoothing:** Prevents zero probabilities for unseen character sequences.
- **Hierarchical backoff:** Combines different n-gram orders using weighted probabilities.
- **Bidirectional context:** Considers both preceding and following known characters when estimating the probability of an unknown character.
- **Word-length conditioning:** Incorporates character frequency distributions specific to the target word length.
- **Log-probability scoring:** Combines probabilities across unknown positions while reducing numerical underflow.

The final decoder evaluates candidate characters using these statistical signals and selects the character with the highest overall score.

## Evaluation Results

All decoding strategies are evaluated using the provided `evaluate_decoder()` function.

The evaluation metric is the **average number of incorrect guesses across the 1,000 evaluation words**. Lower values indicate better performance.

| Question | Decoder | Average Incorrect Guesses |
|:---:|---|---:|
| Q2 | Random Decoder | 16.700 |
| Q3 | Unigram Decoder | 10.358 |
| Q4 | Length-Aware Unigram Decoder | 10.343 |
| Q5 | Bigram Decoder | 8.949 |
| Q6 | Advanced Hybrid N-gram Decoder | **7.152** |


### Results Analysis

The results demonstrate the advantage of incorporating statistical patterns and character context into language modeling.

- **Random Decoder:** Provides a baseline with an average of 16.7 incorrect guesses.
- **Unigram Decoder:** Significantly improves performance by prioritizing frequently occurring characters, reducing the average to 10.358.
- **Length-Aware Unigram:** Achieves a slight improvement over the basic unigram model (10.343), suggesting that word-length conditioning alone provides limited additional benefit.
- **Bigram Decoder:** Reduces the average to 8.949 by incorporating character-level context.
- **Advanced Hybrid N-gram Decoder:** Achieves the best performance, with an average of **7.152 incorrect guesses**.

Overall, the final model reduces the average number of incorrect guesses by approximately **57.2% compared with the random baseline** and **30.95% compared with the basic unigram model**.

These results demonstrate the benefits of combining higher-order n-grams, Laplace smoothing, bidirectional context, and word-length information.

## Dataset

The assignment uses the provided English word bank:

`hw1_word_bank.txt`

No external dataset downloads are required.

## Requirements

- Python 3.11
- Jupyter Notebook or JupyterLab

The implementation relies exclusively on Python's standard library. No additional third-party Python packages are required.

## Key Learning Outcomes

Through this assignment, I explored:

- Implementing character-level language models from scratch.
- Estimating probabilities using unigram and higher-order n-gram statistics.
- Understanding the importance of context in language modeling.
- Handling unseen sequences through Laplace smoothing and backoff.
- Combining forward and backward character context.
- Conditioning character predictions on word length.
- Evaluating NLP models on held-out data.
- Improving prediction strategies through probabilistic modeling.

---

*Part of my Natural Language Processing coursework.*