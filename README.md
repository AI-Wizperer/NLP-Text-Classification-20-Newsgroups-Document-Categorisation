# NLP Text Classification — 20 Newsgroups Document Categorisation

Implementing NLP-based text classification techniques from **Chen (2023)**, *"Ethics and discrimination in artificial intelligence-enabled recruitment practices"* (Humanities and Social Sciences Communications, Nature Portfolio). Four classifiers — **Naive Bayes**, **Logistic Regression**, **Random Forest**, and **Linear SVM** — are compared on the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset) benchmark using TF-IDF feature extraction.

---

## Table of Contents

- [Paper Reference](#paper-reference)
- [Task Overview](#task-overview)
- [Dataset](#dataset)
- [Text Preprocessing Pipeline](#text-preprocessing-pipeline)
- [Feature Extraction](#feature-extraction)
- [Models](#models)
  - [Multinomial Naive Bayes](#multinomial-naive-bayes)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
  - [Linear SVM](#linear-svm)
- [Evaluation Metrics](#evaluation-metrics)
- [Prediction & Inference](#prediction--inference)
- [Getting Started](#getting-started)
- [Notebook Structure](#notebook-structure)
- [Outputs](#outputs)
- [Tech Stack](#tech-stack)
- [License](#license)

---

## Paper Reference

| Field | Details |
|-------|---------|
| **Title** | Ethics and discrimination in artificial intelligence-enabled recruitment practices |
| **Author** | Zhisheng Chen |
| **Journal** | Humanities and Social Sciences Communications (Nature Portfolio) |
| **Impact Factor** | 3.7 (Q1 — Arts & Humanities, Business & Management, Social Sciences) |
| **Year** | 2023 |
| **DOI** | [10.1057/s41599-023-02079-x](https://doi.org/10.1057/s41599-023-02079-x) |

**Key points from the paper:**
1. AI-enabled recruitment systems use NLP to automatically screen and classify documents/resumes
2. These systems extract features from text and apply machine learning classifiers
3. The paper discusses how bias can emerge in automated classification systems
4. Text classification is the core technology behind AI recruitment screening

This notebook replicates the NLP classification methodology described in the paper, using the 20 Newsgroups dataset as a proxy for document categorisation tasks similar to resume screening.

---

## Task Overview

```
┌──────────────────┐      ┌──────────────┐      ┌────────────┐      ┌──────────────────┐
│  Raw Document    │─────▶│  Preprocess   │─────▶│  TF-IDF    │─────▶│  Classifier      │
│  (newsgroup post)│      │  (clean,      │      │  Vectorise │      │  (NB/LR/RF/SVM)  │
│                  │      │   lemmatise)  │      │  (5000 dim)│      │                  │
└──────────────────┘      └──────────────┘      └────────────┘      └──────────────────┘
                                                                          │
                                                                          ▼
                                                                  ┌──────────────────┐
                                                                  │ Predicted Topic  │
                                                                  │ (10 categories)  │
                                                                  └──────────────────┘
```

- **Input:** Raw text content of a newsgroup post
- **Output:** One of 10 topic categories

**Real-world parallel:** This demonstrates the same NLP text classification pipeline used in AI recruitment systems for automated resume/document screening.

---

## Dataset

**[20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)** — a standard benchmark for text classification, built into scikit-learn.

| Property | Value |
|----------|-------|
| Total documents | ~18,000 (full dataset) |
| Categories used | 10 (selected from 20 for clearer demonstration) |
| Features | Raw text content |
| Source | `sklearn.datasets.fetch_20newsgroups` |

### Selected Categories

| Category | Topic |
|----------|-------|
| `comp.graphics` | Computer Graphics |
| `comp.sys.mac.hardware` | Mac Hardware |
| `rec.autos` | Automobiles |
| `rec.sport.baseball` | Baseball |
| `sci.med` | Medicine |
| `sci.space` | Space Science |
| `talk.politics.misc` | Politics |
| `talk.religion.misc` | Religion |
| `misc.forsale` | For Sale |
| `soc.religion.christian` | Christianity |

These 10 categories were chosen for diversity — covering technology, sports, science, politics, and religion — while including deliberately overlapping topics (e.g., `talk.religion.misc` vs `soc.religion.christian`) to simulate real-world classification challenges.

**Metadata removal:** Headers, footers, and quotes are stripped during loading to force models to classify based on content rather than structural metadata.

---

## Text Preprocessing Pipeline

Following the NLP preprocessing approach discussed in Chen (2023):

| Step | Operation | Purpose |
|------|-----------|---------|
| 1 | Lowercase conversion | Normalise case |
| 2 | URL removal | Strip `http://`, `www.` links |
| 3 | Email removal | Strip email addresses |
| 4 | Special character & number removal | Keep only alphabetic tokens |
| 5 | Whitespace normalisation | Collapse multiple spaces |
| 6 | Tokenisation | Split into individual words (NLTK `word_tokenize`) |
| 7 | Stopword removal | Remove English stopwords + tokens ≤ 2 characters |
| 8 | Lemmatisation | Reduce words to base forms (NLTK `WordNetLemmatizer`) |

---

## Feature Extraction

**TF-IDF Vectorisation** with the following configuration:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `max_features` | 5,000 | Maximum vocabulary size |
| `ngram_range` | (1, 2) | Capture unigrams and bigrams |
| `min_df` | 3 | Ignore terms appearing in fewer than 3 documents |
| `max_df` | 0.9 | Ignore terms appearing in more than 90% of documents |
| `sublinear_tf` | True | Apply logarithmic TF scaling (1 + log(tf)) |

This produces a sparse feature matrix of shape `(n_documents, 5000)`.

---

## Models

### Multinomial Naive Bayes

| Parameter | Value |
|-----------|-------|
| `alpha` | 0.1 (Laplace smoothing) |

A probabilistic classifier well-suited for text data with TF-IDF or count-based features. Fast to train and provides a strong baseline for document classification.

### Logistic Regression

| Parameter | Value |
|-----------|-------|
| `C` | 1.0 |
| `solver` | lbfgs |
| `multi_class` | multinomial |
| `max_iter` | 1000 |

Multinomial logistic regression with L2 regularisation. The lbfgs solver handles the multiclass objective natively.

### Random Forest

| Parameter | Value |
|-----------|-------|
| `n_estimators` | 200 |
| `max_depth` | 50 |
| `min_samples_split` | 5 |

An ensemble of 200 decision trees with depth-limited regularisation. Generally less effective on high-dimensional sparse text features than linear models, but included for comparison.

### Linear SVM

| Parameter | Value |
|-----------|-------|
| `C` | 1.0 |
| `max_iter` | 1000 |

Linear Support Vector Classification — typically the strongest performer for text classification due to its effectiveness in high-dimensional sparse feature spaces.

---

## Evaluation Metrics

All four models are evaluated on the held-out test set:

| Metric | Averaging | Description |
|--------|-----------|-------------|
| **Accuracy** | — | Overall fraction of correct predictions |
| **Precision** | Weighted | Proportion of positive predictions that are correct, per class |
| **Recall** | Weighted | Proportion of actual positives correctly identified, per class |
| **F1-Score** | Weighted | Harmonic mean of precision and recall |

### Visualisations

The notebook produces:

- **Category distribution** — bar chart of document counts per category
- **Model comparison** — grouped bar chart of all metrics across four models
- **Confusion matrix** — heatmap for the best-performing model (Linear SVM)
- **Per-class classification report** — precision, recall, F1 for each of the 10 categories
- **Sample predictions** — correct vs misclassified document examples with actual and predicted labels

---

## Prediction & Inference

The notebook includes a reusable function for classifying any new document:

```python
predicted_category = predict_category(
    text="I just upgraded my Mac with a new graphics card and more RAM.",
    model=svm_model,
    vectorizer=tfidf_vectorizer,
    target_names=newsgroups_train.target_names,
)
# Output: *** PREDICTED CATEGORY: comp.sys.mac.hardware ***
#   comp.sys.mac.hardware              : █████████████████████████ 100.0%  ◄── BEST MATCH
#   comp.graphics                      : ██████████               42.1%
#   misc.forsale                       : ████████                 33.7%
#   ...
```

Three example documents are demonstrated:
1. **Technology document** — computer hardware discussion
2. **Sports document** — baseball game recap
3. **Science document** — NASA Mars mission

---

## Getting Started

### Requirements

- **Hardware:** CPU is sufficient — no GPU needed
- **Python:** 3.8+

### Installation

```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk
```

### NLTK Data

The notebook automatically downloads required NLTK resources:

```python
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### Running

1. Open `Newsgroups_REAL_DATA.ipynb` in Jupyter or Google Colab.
2. Run all cells sequentially.
3. The 20 Newsgroups dataset is loaded automatically from scikit-learn — no manual download required.

---

## Notebook Structure

| Cell(s) | Section | Description |
|---------|---------|-------------|
| 0 | Introduction | Paper information, implementation overview |
| 1–2 | Setup | Import libraries (NLP, ML, visualisation) |
| 3–4 | Background | Paper summary, dataset information and justification |
| 5–8 | Data Loading | Load 10 selected categories, display samples, visualise distribution |
| 9–12 | Preprocessing | `preprocess_text()` function — lowercase, URL/email removal, lemmatisation; apply to train/test; show before/after example |
| 13–15 | Feature Extraction | TF-IDF vectorisation (5000 features, unigrams + bigrams), display top features |
| 16–17 | Evaluation Setup | `evaluate_model()` function, results accumulator |
| 18 | Naive Bayes | Train `MultinomialNB(alpha=0.1)`, evaluate |
| 19 | Logistic Regression | Train multinomial LR with lbfgs solver, evaluate |
| 20 | Random Forest | Train 200-tree ensemble, evaluate |
| 21 | Linear SVM | Train `LinearSVC(C=1.0)`, evaluate |
| 22–26 | Results | Summary table, model comparison chart, confusion matrix, classification report |
| 27–33 | Prediction | Sample predictions, correct vs wrong analysis, `predict_category()` function, 3 example documents |
| 34–35 | Final Summary | Paper reference, dataset info, methodology recap, best model results |

---

## Outputs

| Artifact | Description |
|----------|-------------|
| Model comparison table | Accuracy, Precision, Recall, F1 across all four models |
| Category distribution chart | Document counts per newsgroup topic |
| Confusion matrix | 10 × 10 heatmap for best model |
| Classification report | Per-class precision, recall, F1 |
| Model comparison bar chart | Grouped metrics visualisation |

---

## Tech Stack

| Library | Role |
|---------|------|
| [scikit-learn](https://scikit-learn.org/) | Naive Bayes, Logistic Regression, Random Forest, LinearSVC, TF-IDF, evaluation metrics, dataset loading |
| [NLTK](https://www.nltk.org/) | Tokenisation, stopword removal, lemmatisation |
| [Matplotlib / Seaborn](https://matplotlib.org/) | Model comparison charts, confusion matrix, category distribution |
| [NumPy / Pandas](https://numpy.org/) | Data manipulation and results aggregation |

---

## License

This project is for educational and research purposes.
