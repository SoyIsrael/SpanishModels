# Wikibooks Language Classification Experiments

This repository contains a set of small experiments for **binary language classification** (English vs. Spanish) using text from a Wikibooks-style SQLite database.
To download dataset: https://www.kaggle.com/datasets/dhruvildave/wikibooks-dataset

The goal is to compare:
- A simple **Logistic Regression** baseline  
- A few classic ML models (**Logistic Regression, Naive Bayes, SVM**)  
- A **multilingual Transformer (DistilBERT)** fine-tuned on the same task  

All experiments read from the same SQLite database and treat the **page body text** as input and **language** (*english* / *spanish*) as the label.

---

## Repository Structure

- `experiement1.py`  
  Baseline TF–IDF + Logistic Regression model, plus simple feature-importance inspection.

- `experiement2.py`  
  TF–IDF features + comparison of three models:
  - Logistic Regression  
  - Multinomial Naive Bayes  
  - Support Vector Machine (SVM)

- `experiement3.py`  
  Fine-tuning a multilingual DistilBERT model for sequence classification using the Hugging Face `transformers` and `datasets` libraries.

- `utils/data/wikibooks.sqlite`  
  SQLite database containing at least two tables:
  - `en` – English pages  
  - `es` – Spanish pages  

Each table is expected to have the columns:
- `title`
- `body_text`
- `abstract`

---

## Data Assumptions

All scripts assume:

- Database path: `./utils/data/wikibooks.sqlite`
- Two language tables:
  - `en` (English)
  - `es` (Spanish)
- Schema (for both `en` and `es`):

```sql
SELECT title, body_text, abstract FROM en;
SELECT title, body_text, abstract FROM es;
