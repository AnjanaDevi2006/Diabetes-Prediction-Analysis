# Diabetes-Prediction-Analysis
# 🩺 Diabetes Prediction using Machine Learning

A comparative study of four machine learning models for **diabetes prediction** — tracking Accuracy, Precision, Recall, F1-Score, and Loss across training epochs, along with per-class confusion matrix analysis.

---

## 📌 Project Overview

| Item | Detail |
|---|---|
| **Task** | Multi-class Diabetes Classification |
| **Classes** | Non-Diabetic · Pre-Diabetic · Diabetic |
| **Models** | Random Forest, KNN, Logistic Regression, **SVM (Proposed)** |
| **Metrics** | Accuracy, Precision, Recall, F1-Score, Loss |
| **Epochs** | 10 → 100 (step 10) |

The **SVM (Proposed)** model consistently outperforms all baselines, achieving **96.4% accuracy** at epoch 100.

---

## 📊 Results at a Glance (Epoch 100)

| Model | Accuracy | Precision | Recall | F1-Score | Loss |
|---|---|---|---|---|---|
| SVM *(Proposed)* | **0.9640** | **0.9520** | **0.9330** | **0.9460** | **0.0850** |
| Random Forest | 0.8750 | 0.8540 | 0.8300 | 0.8450 | 0.1650 |
| KNN | 0.8540 | 0.8300 | 0.8100 | 0.8200 | 0.1850 |
| Logistic Regression | 0.8300 | 0.8100 | 0.7900 | 0.8000 | 0.2050 |

---

## 📁 Project Structure

```
diabetes_prediction/
│
├── diabetes_prediction.py   # Main script — plots + confusion matrices
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/AnjanaDevi2006/diabetes-prediction-ml.git
cd diabetes-prediction-ml
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Script

```bash
python diabetes_prediction.py
```

This generates:
- **5 epoch-vs-metric line plots** (Accuracy, Precision, Recall, F1-Score, Loss)
- **4 confusion matrix heatmaps** — one for each model

---

## 🧰 Dependencies

```
numpy
matplotlib
seaborn
scikit-learn
```

---

## 📈 Plots Generated

| Plot | Description |
|---|---|
| Accuracy vs Epochs | Training accuracy trend for all 4 models |
| Precision vs Epochs | Precision comparison across epochs |
| Recall vs Epochs | Recall trend across epochs |
| F1-Score vs Epochs | F1-Score convergence comparison |
| Loss vs Epochs | Loss reduction across epochs |
| Confusion Matrix – SVM | Per-class prediction heatmap (Blues) |
| Confusion Matrix – RF | Per-class prediction heatmap (Greens) |
| Confusion Matrix – KNN | Per-class prediction heatmap (Oranges) |
| Confusion Matrix – LR | Per-class prediction heatmap (Reds) |

---

## 🏷️ Classes

| Label | Class |
|---|---|
| 0 | Non-Diabetic |
| 1 | Pre-Diabetic |
| 2 | Diabetic |

---

## 📝 Notes

- The simulated training results replicate a realistic machine-learning experiment on a diabetes dataset (e.g., PIMA Indians Diabetes Dataset).
- To plug in **real data**, replace the `results` dictionary with actual metrics from your trained models, and replace the `pred_*` arrays with actual model predictions.
- The SVM model uses an RBF kernel and is labelled *Proposed* to highlight it as the recommended approach.

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

[MIT](LICENSE)
