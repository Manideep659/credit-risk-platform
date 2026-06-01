import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive Agg backend to prevent GUI warnings when running in background
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

def get_explainer_and_values(model_path="models/model.pkl", data_path="data/processed_train.csv", sample_size=500):
    """
    Loads model and representative dataset sample, then computes SHAP TreeExplainer and values.
    Uses sample_size=500 by default for speed and production-grade reliability.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # Load model
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)

    # Load data
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Separate features (dropping target and applicant key)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])

    # Take a representative stratified-like sample for explainability performance
    print(f"Taking a representative sample of {sample_size} records...")
    X_sample = X.sample(n=sample_size, random_state=42)

    # 2. Generate SHAP TreeExplainer
    print("Initialising SHAP TreeExplainer for LightGBM...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    print("Computing SHAP values for the sample...")
    shap_values = explainer(X_sample)
    
    return model, X, X_sample, explainer, shap_values

def plot_shap_summary(shap_values, X_sample, save_dir="documents/shap"):
    """
    Generates and saves the SHAP Summary Plot (dot plot showing impact of feature values).
    Compatible with Streamlit by returning the Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # We pass the Explanation slice for binary positive class if multi-dimensional
    if len(shap_values.shape) == 3:  # (samples, features, classes)
        # Binary prediction (extracting class 1)
        exp_positive = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X_sample.values,
            feature_names=X_sample.columns
        )
    else:
        exp_positive = shap_values

    shap.plots.beeswarm(exp_positive, max_display=15, show=False)
    plt.title("SHAP Global Summary Beeswarm Plot", fontsize=14, pad=15)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "shap_summary_plot.png")
        print(f"Saving SHAP Summary Plot to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_shap_importance(shap_values, X_sample, save_dir="documents/shap"):
    """
    Generates and saves the SHAP Feature Importance Plot (bar plot of mean absolute SHAP).
    Compatible with Streamlit by returning the Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(shap_values.shape) == 3:
        exp_positive = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=X_sample.values,
            feature_names=X_sample.columns
        )
    else:
        exp_positive = shap_values

    shap.plots.bar(exp_positive, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (Mean |SHAP Value|)", fontsize=14, pad=15)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "shap_importance_plot.png")
        print(f"Saving SHAP Importance Plot to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def plot_shap_waterfall(shap_values, X_sample, sample_idx=0, save_dir="documents/shap"):
    """
    Generates and saves the SHAP Waterfall Plot for a single customer sample.
    Compatible with Streamlit by returning the Figure object.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    if len(shap_values.shape) == 3:
        single_val = shap.Explanation(
            values=shap_values.values[sample_idx, :, 1],
            base_values=shap_values.base_values[sample_idx, 1],
            data=X_sample.iloc[sample_idx].values,
            feature_names=X_sample.columns
        )
    else:
        single_val = shap_values[sample_idx]

    shap.plots.waterfall(single_val, max_display=12, show=False)
    plt.title(f"SHAP Waterfall Plot for Applicant (Index {sample_idx})", fontsize=14, pad=15)
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"shap_waterfall_applicant_{sample_idx}.png")
        print(f"Saving SHAP Waterfall Plot to {save_path}...")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig

def get_top_influential_features(shap_values, X_sample):
    """
    Computes and returns the top 20 most influential features based on mean absolute SHAP value.
    """
    if len(shap_values.shape) == 3:
        vals = shap_values.values[:, :, 1]
    else:
        vals = shap_values.values
        
    mean_abs_shap = np.mean(np.abs(vals), axis=0)
    importance_df = pd.DataFrame({
        'Feature': X_sample.columns,
        'Mean_Absolute_SHAP': mean_abs_shap
    }).sort_values(by='Mean_Absolute_SHAP', ascending=False)
    
    return importance_df.head(20)

def main():
    """
    Executes the standard pipeline: loads artifacts, computes SHAP, prints top 20 features,
    and exports Summary, Importance, and single-applicant Waterfall plots.
    """
    # 1. Load model, data, and compute values
    model, X, X_sample, explainer, shap_values = get_explainer_and_values()
    
    # 4. Save directory
    save_dir = "documents/shap"
    
    # 3. Create and Save Summary Beeswarm Plot
    plot_shap_summary(shap_values, X_sample, save_dir)
    
    # 3. Create and Save Importance Bar Plot
    plot_shap_importance(shap_values, X_sample, save_dir)
    
    # 3. Create and Save Waterfall Plot for the first applicant in sample (index 0)
    plot_shap_waterfall(shap_values, X_sample, sample_idx=0, save_dir=save_dir)
    
    # 5. Print top 20 most influential features
    top_20 = get_top_influential_features(shap_values, X_sample)
    
    print("\n" + "="*50)
    print("TOP 20 MOST INFLUENTIAL FEATURES (SHAP GLOBAL)")
    print("="*50)
    for idx, row in enumerate(top_20.itertuples(), 1):
        print(f"{idx:02d}. {row.Feature:<30} | Mean |SHAP| Impact: {row.Mean_Absolute_SHAP:.6f}")
    print("="*50 + "\n")
    print(f"[SUCCESS] SHAP explainability analysis complete. Visualizations saved inside: {save_dir}/")

if __name__ == "__main__":
    main()
