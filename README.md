# 🚀 Mamba for Sequential Event Recommendation in Event-Based Social Networks

> Exploring the effectiveness of Mamba State Space Models for sequential event recommendation in Event-Based Social Networks (EBSNs).

---

## 🌟 Overview

This project investigates whether the recently proposed **Mamba State Space Model (SSM)** can effectively capture long-range dependencies in user-event interaction sequences and improve recommendation performance.

---

## 🎯 Task

Given:

- 👤 User
- 🎪 Target Event
- 📜 Historical Interaction Sequence

Predict:

> The probability that the user will be interested in the target event.

---

## 🏗️ Data Processing Pipeline

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4f2ccf6-c46c-40f6-aae6-827ff4dfdb43" width="900">
</p>

## 🧠 Model Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/3ee5614a-8c23-49fa-8ca8-9555d703ab0a" width="500">
</p>

The model consists of three major components:

### 📚 History Encoder

- Sequential user-event interactions
- Positional Encoding
- Residual Mamba Blocks
- Long-range dependency modeling

### 🎪 Target Event Encoder

- Multi-Layer Perceptron (MLP)
- Event feature representation

### 👤 User Encoder

- Learnable user embeddings

### 🎯 Prediction Layer

Concatenate all representations and perform binary classification:

```
P(User Interested | History, User, Target Event)
```

---

## 📦 Installation

Install required packages:

```bash
pip install torch pandas mamba-ssm
```

---

## 🚀 Quick Start

### Step 1: Generate Predictions

```bash
python data_processor.py
```

Output:

```text
prediction_results.csv
```

containing:

| Column | Description |
|----------|------------|
| user_id | User ID |
| event_id | Event ID |
| probability | Probability of Intereted |
| prediction | Interested or Not |
| actual_label | Ground Truth |

---

### Step 2: Evaluate Performance

Modify:

```python
evaluation.py
```

- Input prediction file
- Evaluation metrics

Then run:

```bash
python evaluation.py
```

---

## 📊 Evaluation Metrics

### Classification Metrics

- Precision
- Recall
- F1-score

### Ranking Metrics

- MAP@K
- MRR@K
- NDCG@K
- Recall@K

---

## 🗂 Dataset

This implementation is evaluated on the:

🏆 **Kaggle Event Recommendation Engine Challenge Dataset**

Statistics:

| Item | Value |
|--------|--------|
| Users | 2,034 |
| Events | 8,846 |
| Interactions | 15,398 |
| Positive Ratio | 26.83% |
| Negative Ratio | 73.17% |
---

## 🔬 Future Directions

Potential extensions include:

- Graph-enhanced Mamba
- Social-aware Sequential Recommendation
- Temporal Graph Neural Networks
- Hybrid Mamba-GNN Architectures
- Large Language Model Enhanced Event Recommendation

---

## 👨‍💻 Author

**Delin Meng**

🎓 Chungbuk National University

📧 dylanmeng@chungbuk.ac.kr

---

## 📬 Contact

Feel free to open an issue or contact me via email for questions, suggestions, or collaborations.

📧 dylanmeng@chungbuk.ac.kr
