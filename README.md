# Credit Card Fraud Detection using CatBoost & SMOTE

## 📌 Project Overview
This project implements a high-performance machine learning solution to identify fraudulent credit card transactions. Recognizing that fraud detection is a classic imbalanced classification problem, this implementation utilizes **SMOTE (Synthetic Minority Over-sampling Technique)** for data balancing and **CatBoost** for robust gradient boosting classification.

## 🛠️ Technical Stack
*   **Languages:** Python
*   **Core Libraries:** Pandas, CatBoost, Scikit-learn, Imbalanced-learn (SMOTE)
*   **Techniques:** Gradient Boosting, Synthetic Oversampling, Feature Selection

## 📂 Dataset Features
The model processes 29 features to distinguish between legitimate and fraudulent activity:
*   **V1 - V28:** PCA-transformed features representing hidden transaction patterns.
*   **Amount:** The numerical value of the transaction.
*   **Class (Target):** The binary classification target (0: Legitimate, 1: Fraudulent).

## 🚀 Key Features
*   **Imbalance Handling:** Utilizes **SMOTE** to generate synthetic samples for the minority class, preventing model bias.
*   **Stratified Splitting:** Employs stratified train-test splits to maintain identical class distributions across both sets.
*   **Optimized Training:** Configured with **Early Stopping** (50 rounds) and **AUC as the evaluation metric** to ensure peak performance without overfitting.
*   **Persistence:** Automatically exports the trained model as a `.cbm` file for future inference.

## 💻 Installation & Usage
1. **Clone the repository and install dependencies:**
   ```bash
   pip install pandas catboost scikit-learn imbalanced-learn
   Prepare Data:
Ensure your dataset is saved as compressed_data.csv in the root directory.

Execute:
python main.py

📈 Evaluation Results
The script outputs a full classification report including:
Accuracy Score
Precision, Recall, and F1-Score (Crucial for imbalanced fraud data)
Final Saved Model: catboost_creditcardfraud.cbm

Precision, Recall, and F1-Score (Crucial for imbalanced fraud data)

Final Saved Model: catboost_creditcardfraud.cbm
