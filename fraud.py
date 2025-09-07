import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load dataset (replace with your filename)
    df = pd.read_csv('compressed_data.csv')
    # Group by the target column 'Class' and count samples in each group
    class_counts = df.groupby('Class').size()
    print(class_counts)
    # Use PCA features V1 to V28 plus Amount
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['Amount']
    target_col = 'Class'

    X = df[feature_cols]
    y = df[target_col]

    # Split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # CatBoost Pools (all numeric features)
    train_pool = Pool(X_train_res, y_train_res)
    test_pool = Pool(X_test, y_test)

    # Initialize model
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.1,
        depth=6,
        eval_metric='AUC',
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50
    )

    # Train model
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    # Evaluate
    y_pred = model.predict(test_pool)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # Save model
    model.save_model('catboost_creditcardfraud.cbm')
    print("Model saved as catboost_creditcardfraud.cbm")

if __name__ == '__main__':
    main()
