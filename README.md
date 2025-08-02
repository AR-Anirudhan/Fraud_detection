# 🕵️‍♀️ Fraud Detection with Machine Learning

This repository contains a Jupyter Notebook for detecting fraudulent credit card transactions using machine learning techniques. The goal is to identify patterns and build models that can accurately flag suspicious activity.

## 📁 Files

- `fraud_detection.ipynb`: The main notebook that walks through data preprocessing, exploratory data analysis, model training, and evaluation.

## 📊 Dataset

The dataset used is a publicly available credit card fraud dataset, typically sourced from [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

- **Features**: 30 anonymized numerical variables (V1–V28, Time, Amount)  
- **Target**: `Class` (0 = legitimate, 1 = fraud)  
- **Note**: The dataset is highly imbalanced (~0.17% fraud cases)

## 🛠️ Technologies Used

- Python 3.x  
- Jupyter Notebook  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- (Optional) XGBoost / LightGBM

## 🧠 ML Models Used

- Logistic Regression  
- Random Forest  
- (Optional) Gradient Boosting / XGBoost  
- Evaluation metrics: Precision, Recall, F1-score, ROC AUC

Here are the instructions on how to launch and run the Jupyter Notebook, formatted in Markdown.

---

## 🚀 Getting Started

To get started with this project, you'll need to have Python and Jupyter Notebook installed on your system.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Launch Jupyter Notebook:**
    Run the following command in your terminal from the project's root directory:
    ```bash
    jupyter notebook
    ```
    This will open a new tab in your web browser showing the Jupyter file browser.

3.  **Run the notebook:**
    Click on `fraud_detection.ipynb` to open it. Once the notebook is open, you can run through the cells to see the analysis and model training process. You can run all cells at once by navigating to `Cell > Run All` in the menu bar.

## 📈 Results

Performance metrics are evaluated on an imbalanced dataset. The emphasis is on **Recall** and **AUC** to maximize the detection of fraudulent transactions.

Models are assessed using the following metrics:
* Precision
* Recall
* F1-Score
* ROC AUC

Visualizations included:
* Confusion Matrix
* ROC Curve
* Feature Importance Plot

---

## ⚖️ License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

## 🙌 Acknowledgments

* [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Scikit-learn documentation and the open-source community

