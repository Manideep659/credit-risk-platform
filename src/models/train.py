import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import lightgbm as lgb

def train_pipeline(data_path="data/processed_train.csv", model_dir="models"):
    """
    Production-grade model training pipeline.
    Loads processed data, splits into stratified train/test sets, trains LightGBM
    with class imbalance handling, evaluates, generates feature importances,
    and saves the model and metrics.
    """
    print(f"Loading preprocessed dataset: {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Separate TARGET from features
    print("Separating features and target variable...")
    # Drop unique identifier SK_ID_CURR as it is not a predictive feature
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    # 3. Split data (80% train, 20% test)
    print("Splitting data into train and test sets (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Essential to preserve the imbalanced class ratio
    )
    
    print(f"Train set shape: {X_train.shape[0]:,} rows | Test set shape: {X_test.shape[0]:,} rows")
    
    # 5. Handle class imbalance using scale_pos_weight
    # Formula: negative_samples / positive_samples
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = float(neg_count / pos_count)
    print(f"Class counts in training: Negative (0) = {neg_count:,}, Positive (1) = {pos_count:,}")
    print(f"Computed scale_pos_weight: {scale_pos_weight:.4f}")
    
    # 4. Train a LightGBM classifier
    print("Initializing and training LightGBM Classifier...")
    clf = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=150,      # High capacity tree framework
        learning_rate=0.05,     # Robust, slow-learning rate
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1,              # Utilize all CPU cores
        verbose=-1              # Quiet execution logs
    )
    
    clf.fit(X_train, y_train)
    print("Model training completed successfully.")
    
    # 6. Evaluate metrics
    print("Evaluating model performance on test set...")
    # Predicted probabilities for ROC AUC
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    # Binary predictions for Precision, Recall, F1
    y_pred_class = clf.predict(X_test)
    
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "precision": float(precision_score(y_test, y_pred_class)),
        "recall": float(recall_score(y_test, y_pred_class)),
        "f1_score": float(f1_score(y_test, y_pred_class))
    }
    
    # 7. Generate feature importance ranking
    print("Extracting feature importances...")
    # Gain-based importance is more indicative of contribution to predictive splits
    importance_values = clf.booster_.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance_values
    }).sort_values(by='Importance', ascending=False)
    
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # 8. Save model and metrics
    model_path = os.path.join(model_dir, "model.pkl")
    metrics_path = os.path.join(model_dir, "metrics.json")
    
    print(f"Saving model to {model_path}...")
    joblib.dump(clf, model_path)
    
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
        
    # 9. Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION SUMMARY (TEST SET)")
    print("="*50)
    print(f"ROC AUC Score: {metrics['roc_auc']:.6f}")
    print(f"Precision:     {metrics['precision']:.6f}")
    print(f"Recall:        {metrics['recall']:.6f}")
    print(f"F1 Score:      {metrics['f1_score']:.6f}")
    print("="*50)
    
    print("\n" + "="*50)
    print("TOP 20 IMPORTANT FEATURES")
    print("="*50)
    top_20 = importance_df.head(20)
    for idx, row in enumerate(top_20.itertuples(), 1):
        # Format names and values nicely
        print(f"{idx:02d}. {row.Feature:<30} | Gain Contribution: {row.Importance:,.2f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_pipeline()
