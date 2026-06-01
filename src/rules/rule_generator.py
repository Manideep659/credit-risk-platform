import os
import pandas as pd
import numpy as np
import matplotlib
# Use non-interactive Agg backend to prevent GUI warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

def generate_rules_and_tree(data_path="data/processed_train.csv", doc_dir="documents/rules"):
    """
    Trains a DecisionTreeClassifier of depth 3 to model borrowers defaults,
    extracts explicit human-readable business rules, exports them to a text file,
    and saves a professional matplotlib visualisation of the decision tree.
    """
    print(f"Loading preprocessed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # 2. Separate TARGET from features
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    # 1. Train a DecisionTreeClassifier(max_depth=3)
    print("Training DecisionTreeClassifier (max_depth=3)...")
    clf = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    print("Decision Tree training complete.")
    
    # Traverse tree to extract rules
    tree_ = clf.tree_
    feature_names = list(X.columns)
    rules = []
    
    # Base baseline default rate for classification reference
    baseline_default = y.mean()
    print(f"Dataset baseline default rate: {baseline_default * 100:.4f}%")
    
    def recurse(node, depth, path):
        # Check if internal node or leaf
        if tree_.feature[node] != -2:
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            
            # Left child (less than or equal to threshold)
            left_path = path + [f"{name} <= {threshold:.4f}"]
            recurse(tree_.children_left[node], depth + 1, left_path)
            
            # Right child (greater than threshold)
            right_path = path + [f"{name} > {threshold:.4f}"]
            recurse(tree_.children_right[node], depth + 1, right_path)
        else:
            # Leaf node reached: extract distribution of classes
            # Locate samples in this leaf
            node_indicator = clf.decision_path(X)
            leaf_indices = clf.apply(X)
            in_leaf = (leaf_indices == node)
            leaf_y = y[in_leaf]
            
            total_samples = len(leaf_y)
            default_count = int(leaf_y.sum())
            default_rate = default_count / total_samples if total_samples > 0 else 0.0
            
            # Determine Risk Label compared to baseline (8.07%)
            risk_label = "High Risk" if default_rate > baseline_default else "Low Risk"
            
            # Construct human-readable rule text
            if len(path) == 0:
                rule_text = f"IF Root THEN {risk_label}"
            else:
                rule_text = "IF " + " \nAND ".join(path) + f" \nTHEN {risk_label} (Default Rate: {default_rate * 100:.2f}%, n={total_samples:,})"
            
            rules.append({
                'rule': rule_text,
                'default_rate': default_rate,
                'samples': total_samples
            })

    print("Extracting business rules from tree nodes...")
    recurse(0, 1, [])
    
    # Sort rules: Highest risk rules first
    rules_sorted = sorted(rules, key=lambda x: x['default_rate'], reverse=True)
    
    # Ensure export path exists
    os.makedirs(doc_dir, exist_ok=True)
    rules_txt_path = os.path.join(doc_dir, "business_rules.txt")
    
    # 3. Export rules to documents/rules/business_rules.txt
    print(f"Exporting human-readable business rules to {rules_txt_path}...")
    with open(rules_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("CREDIT RISK AUTOMATED BUSINESS RULES (DECISION TREE EXPLANATIONS)\n")
        f.write("="*80 + "\n")
        f.write(f"Baseline Platform Default Rate: {baseline_default * 100:.2f}%\n")
        f.write(f"Total Rules Generated: {len(rules_sorted)}\n\n")
        
        for idx, r in enumerate(rules_sorted, 1):
            f.write(f"Rule #{idx:02d} (Risk Tier: {'HIGH' if r['default_rate'] > baseline_default else 'LOW'}):\n")
            f.write(r['rule'] + "\n")
            f.write("-"*50 + "\n\n")
    
    # 4. Print top rules
    print("\n" + "="*50)
    print("EXTRACTED DECISION TREE BUSINESS RULES (SORTED BY RISK)")
    print("="*50)
    for idx, r in enumerate(rules_sorted, 1):
        print(f"\nRule #{idx:02d} (Default Rate: {r['default_rate']*100:.2f}%, Count: {r['samples']:,}):")
        print(r['rule'])
    print("="*50 + "\n")
    
    # 5. Create visual representation of the decision tree
    print("Generating decision tree visualisation...")
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=X.columns,
        class_names=["Repayed (0)", "Defaulted (1)"],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=3,
        precision=2,
        ax=ax
    )
    plt.title("Credit Risk Rule Generator Decision Tree Structure (Depth=3)", fontsize=16, pad=15)
    
    # 6. Save tree visualization
    tree_img_path = os.path.join(doc_dir, "tree.png")
    print(f"Saving decision tree visualisation to {tree_img_path}...")
    plt.savefig(tree_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Automated business rule generation complete! Rules & Visuals saved in: {doc_dir}/")

if __name__ == "__main__":
    generate_rules_and_tree()
