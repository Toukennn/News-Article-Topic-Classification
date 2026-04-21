# 📰 News Article Topic Classification

A machine learning project for **automatic classification of news articles** into predefined categories using **textual content and metadata**.

---

## 📌 Overview

With the massive growth of online news, organizing articles into meaningful categories has become essential for:

- Content recommendation systems  
- News aggregation platforms  
- Targeted advertising  

This project builds a **multiclass classification pipeline** to automatically assign each article to one of 7 categories:

| Label | Category |
|------|--------|
| 0 | International News |
| 1 | Business |
| 2 | Technology |
| 3 | Entertainment |
| 4 | Sports |
| 5 | General News |
| 6 | Health |

---

## 📂 Dataset

- ~100,000 news articles  
- Features include:
  - `title`
  - `article`
  - `source`
  - `page_rank`
  - `timestamp`
  - `label` (only in development set)

### Data Split
- **Development set** → training & validation  
- **Evaluation set** → prediction only  

---

## ⚙️ Methodology

### 1. Preprocessing

- Combined text fields:
  - Article body (base)
  - Title (weighted ×2)
  - Source (weighted ×3)
- Lowercasing and missing value handling
- Metadata features:
  - Cyclic encoding of time (sin/cos)
  - Page rank

---

### 2. Feature Engineering

- **TF-IDF vectorization**
  - Word-level n-grams
  - Character-level n-grams

Final feature space:
- ~65,000 features
- Highly sparse

---

### 3. Model

Final model:

**TF-IDF + Linear Support Vector Classifier (LinearSVC)**

Why?

- Efficient on high-dimensional sparse data  
- Strong generalization  
- Scalable  

---

### 4. Hyperparameter Tuning

- Regularization: `C ∈ [0.09 – 0.20]`
- Class weights: balanced vs custom (final choice)
- Max iterations: up to 1000  

Final configuration:
- `C ≈ 0.095`
- Custom class weights
- `max_iter = 700`
- `dual = False`

---

## 📊 Results

| Model | Macro F1 |
|------|--------|
| **TF-IDF + LinearSVC (Final)** | **0.731** |
| Logistic Regression | ~0.71 |
| LinearSVC + SGD Ensemble | ~0.71 |
| TF-IDF + SVD + XGBoost | ~0.70 |

### Cross-validation
- Mean Macro F1: **0.720**
- Std: **0.0018**

---

## 🧠 Key Insights

- Linear models perform very well on sparse text data  
- Character n-grams improve robustness  
- Metadata significantly boosts performance  
- Simpler models + strong feature engineering outperform complex pipelines  

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Toukennn/News-Article-Topic-Classification.git
cd News-Article-Topic-Classification
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the notebook
Open:
winter_project.ipynb
### 4. Generate predictions

Output format:

Id,Predicted
0,2
1,5
...

---
## 🛠️ Tech Stack
- Python (Jupyter Notebook) 
- Scikit-learn
- Pandas
- NumPy
- Matplotlib 
- Seaborn 
---
## 📁 Repository Structure
```text
├── LISENCE
├── .gitignore
├── main.py
├── news_classification.ipynb
├── report_project.pdf
├── README.md
└── requirements.txt
```
---
## 🔮 Future Improvements
- Use transformer-based models (BERT, RoBERTa) 
- Improve handling of class imbalance
- More advanced feature engineering
- Hybrid or ensemble approaches
--- 
## 👨‍💻 Authors
- Amir Reza Khatibi
- Mehdi Bigdeli
--- 
## ⭐ Final Note 
This project shows that:
Well-engineered classical machine learning pipelines can still achieve strong performance in NLP tasks.
