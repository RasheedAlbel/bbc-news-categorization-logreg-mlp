# Automated News Categorization using Logistic Regression and MLP

> Classifying BBC news articles into 5 topic categories using traditional ML and neural network approaches.

**Course:** Predictive Modeling for Text | **Date:** November 2025  
**Authors:** Rasheed Albel, Christian Paraan, Ada Plata

---

## Overview

News platforms publish thousands of articles daily, making manual categorization
impossible at scale. This project builds an automated text classification system
that assigns BBC news articles to one of five predefined categories:
**Business, Entertainment, Politics, Sport, and Tech**.

Two models are compared — a Multinomial Logistic Regression baseline and a
Multilayer Perceptron (MLP) — both trained on TF-IDF feature representations.

---

## Dataset

- **Source:** BBC News Dataset
- **Size:** ~2,225 articles across 5 categories
- **Split:** 80% train / 20% test (stratified)

| Category      | Sample Count |
|---------------|-------------|
| Business      | 510         |
| Entertainment | 386         |
| Politics      | 417         |
| Sport         | 511         |
| Tech          | 401         |

---

## Methodology

### Data Preprocessing
- Tokenization, Lemmatization, Stopword Removal
- TF-IDF Vectorization with unigrams + bigrams (`ngram_range=(1,2)`)
- Top 5,000 features selected (LogReg) / 20,000 features (MLP)

### Models

**Multinomial Logistic Regression**
- Hyperparameter tuning via GridSearchCV (5-fold stratified CV, 96 combinations)
- Best config: `C=1`, `penalty=l2`, `solver=saga`, `max_iter=100`

**Multilayer Perceptron (MLP)**
- Architecture: Input → 256 → 128 → 5 (Softmax)
- ReLU activations, Dropout (0.3), Adam optimizer
- Best config: `lr=0.002`, `weight_decay=1e-5`, `batch_size=128`

---

## Results

| Model                     | Test Accuracy | Macro F1 |
|---------------------------|--------------|----------|
| Logistic Regression       | **97.98%**   | 0.98     |
| Multilayer Perceptron     | **98.43%**   | 0.9832   |

**MLP is the better-performing model**, generalizing slightly better and handling
nonlinear feature interactions in the larger 20,000-feature space.

### Per-class Performance (MLP)

| Category      | Precision | Recall | F1   |
|---------------|-----------|--------|------|
| Business      | 0.98      | 0.99   | 0.99 |
| Entertainment | 0.95      | 0.99   | 0.97 |
| Politics      | 1.00      | 0.98   | 0.99 |
| Sport         | 1.00      | 1.00   | 1.00 |
| Tech          | 0.99      | 0.96   | 0.97 |

---

## Model Explainability

**SHAP** was used to interpret the Logistic Regression model's predictions.

Top predictive features per class:
- **Business:** company, bank, firm, share, market
- **Entertainment:** film, star, music, band, singer
- **Politics:** party, mr, labour, government, minister
- **Sport:** win, match, club, player, coach
- **Tech:** game, computer, technology, software, user

**Permutation Feature Importance** was used for the MLP, revealing distributed
importance (Δaccuracy ≈ 0.0056 per feature) — indicating the model learned
generalizable patterns rather than overfitting to specific tokens.

---

## Limitations

- Dataset limited to ~2,225 samples from BBC — may not generalize to other news sources
- MLP probability outputs are **overconfident** (calibration curves deviate from the 45° diagonal)
- Out-of-domain text (e.g., social media) may degrade performance

---

## Future Work

- Expand to larger, noisier datasets (real-time scraped news, social media)
- Improve probability calibration via temperature scaling or Platt scaling
- Explore transformer-based models (BERT, DistilBERT)

---

## Tech Stack

`Python` · `scikit-learn` · `PyTorch` · `NLTK` · `SHAP` · `Matplotlib` · `Pandas` · `NumPy`

---

## References 

- Majid, A. (2023, November 2). At 1,500 stories per day, Mail Online is UK’s most prolific news website. Press Gazette. https://pressgazette.co.uk/media-audience-and-business-data/at-1500-stories-per-day-mail-onl ine-is-uks-most-prolific-news-website/
- Yufeng. (2018). BBC articles fulltext and category. Kaggle.com.
https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category

---

## Acknowledgment of Tools

AI tools (e.g., ClaudeAI) were used selectively to support code debugging and structure refinement; all analytical design, methodology selection, and interpretation were completed independently.

---

## Individual Contributions

This was a collaborative project completed as part of MATH 103.1.

My primary contributions included:
- Implementing TF-IDF preprocessing pipeline
- Training and tuning Logistic Regression baseline (GridSearchCV)
- Conducting model evaluation using accuracy, macro-F1, and confusion matrices
- Performing model interpretability analysis using SHAP
- Writing technical documentation and results interpretation
