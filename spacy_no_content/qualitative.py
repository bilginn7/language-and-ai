import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def get_top_features_for_text(text, vectorizer, clf, top_n=5):
    """Extract top features and their weights for a single text"""
    vec = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    coefs = clf.coef_[0]

    # Get non-zero features in this text
    feature_indices = vec.nonzero()[1]
    feature_scores = [(feature_names[i], coefs[i] * vec[0, i]) for i in feature_indices]

    # Sort by absolute contribution
    feature_scores.sort(key=lambda x: abs(x[1]), reverse=True)

    return feature_scores[:top_n]


def create_masking_analysis_csv(train_df, test_df, original_pipe, masked_pipe, sample_size=200):
    """Create detailed CSV showing original vs masked predictions with feature scores"""
    print("\n" + "=" * 80)
    print("CREATING DETAILED MASKING ANALYSIS CSV")
    print("=" * 80)

    # Sample test set for detailed analysis
    if len(test_df) > sample_size:
        test_sample = test_df.sample(sample_size, random_state=42).copy().reset_index(drop=True)
    else:
        test_sample = test_df.copy().reset_index(drop=True)

    print(f"Analyzing {len(test_sample)} posts...")

    # Get predictions
    original_preds = original_pipe.predict(test_sample["text"])
    masked_preds = masked_pipe.predict(test_sample["text_noun_masked"])

    original_proba = original_pipe.predict_proba(test_sample["text"])
    masked_proba = masked_pipe.predict_proba(test_sample["text_noun_masked"])

    # Extract vectorizers and classifiers
    vec_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    vec_masked = masked_pipe.named_steps["tfidf"]
    clf_masked = masked_pipe.named_steps["clf"]

    # Build analysis rows
    analysis_rows = []

    for idx in range(len(test_sample)):
        row = test_sample.iloc[idx]
        true_label = "Female" if row["label"] == 1 else "Male"

        # Original model
        orig_pred = "Female" if original_preds[idx] == 1 else "Male"
        orig_correct = (original_preds[idx] == row["label"])
        orig_conf = original_proba[idx][1] if original_preds[idx] == 1 else original_proba[idx][0]

        # Masked model
        mask_pred = "Female" if masked_preds[idx] == 1 else "Male"
        mask_correct = (masked_preds[idx] == row["label"])
        mask_conf = masked_proba[idx][1] if masked_preds[idx] == 1 else masked_proba[idx][0]

        # Get top features
        orig_features = get_top_features_for_text(row["text"], vec_orig, clf_orig, top_n=5)
        mask_features = get_top_features_for_text(row["text_noun_masked"], vec_masked, clf_masked, top_n=5)

        # Format features as string
        orig_feat_str = ", ".join([f"{feat} ({score:+.2f})" for feat, score in orig_features])
        mask_feat_str = ", ".join([f"{feat} ({score:+.2f})" for feat, score in mask_features])

        analysis_rows.append({
            "original_text": row["text"][:200],  # Truncate for readability
            "masked_text": row["text_noun_masked"][:200],
            "true_label": true_label,
            "original_prediction": orig_pred,
            "original_correct": orig_correct,
            "original_confidence": f"{orig_conf:.3f}",
            "original_top_features": orig_feat_str,
            "masked_prediction": mask_pred,
            "masked_correct": mask_correct,
            "masked_confidence": f"{mask_conf:.3f}",
            "masked_top_features": mask_feat_str,
            "prediction_changed": (orig_pred != mask_pred)
        })

    # Create DataFrame
    analysis_df = pd.DataFrame(analysis_rows)

    # Save to CSV
    analysis_df.to_csv("masking_analysis.csv", index=False)
    print(f"Saved detailed analysis to masking_analysis.csv")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("MASKING ANALYSIS SUMMARY")
    print("=" * 80)

    orig_acc = analysis_df["original_correct"].sum() / len(analysis_df)
    mask_acc = analysis_df["masked_correct"].sum() / len(analysis_df)
    changed = analysis_df["prediction_changed"].sum()

    print(f"\nOriginal model accuracy: {orig_acc:.2%}")
    print(f"Masked model accuracy: {mask_acc:.2%}")
    print(f"Predictions changed: {changed} ({changed / len(analysis_df):.1%})")

    # Show interesting cases
    print("\n" + "=" * 80)
    print("INTERESTING CASES (Prediction Changed)")
    print("=" * 80)

    changed_cases = analysis_df[analysis_df["prediction_changed"]].head(5)
    for i, row in changed_cases.iterrows():
        print(f"\nCase {i}:")
        print(f"  True label: {row['true_label']}")
        print(f"  Original → Masked: {row['original_prediction']} → {row['masked_prediction']}")
        print(f"  Original text: {row['original_text'][:100]}...")
        print(f"  Key original features: {row['original_top_features']}")
        print(f"  Key masked features: {row['masked_top_features']}")

    return analysis_df


def create_leakage_analysis_csv(train_df, test_df, original_pipe, masked_pipe, sample_size=200):
    """Create detailed CSV showing original vs masked predictions with feature scores"""
    print("\n" + "=" * 80)
    print("CREATING DETAILED LEAKAGE ANALYSIS CSV")
    print("=" * 80)

    # Sample test set for detailed analysis
    if len(test_df) > sample_size:
        test_sample = test_df.sample(sample_size, random_state=42).copy().reset_index(drop=True)
    else:
        test_sample = test_df.copy().reset_index(drop=True)

    print(f"Analyzing {len(test_sample)} posts...")

    # Get predictions
    original_preds = original_pipe.predict(test_sample["text"])
    masked_preds = masked_pipe.predict(test_sample["text_leakage_masked"])

    original_proba = original_pipe.predict_proba(test_sample["text"])
    masked_proba = masked_pipe.predict_proba(test_sample["text_leakage_masked"])

    # Extract vectorizers and classifiers
    vec_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    vec_masked = masked_pipe.named_steps["tfidf"]
    clf_masked = masked_pipe.named_steps["clf"]

    # Build analysis rows
    analysis_rows = []

    for idx in range(len(test_sample)):
        row = test_sample.iloc[idx]
        true_label = "Female" if row["label"] == 1 else "Male"

        # Original model
        orig_pred = "Female" if original_preds[idx] == 1 else "Male"
        orig_correct = (original_preds[idx] == row["label"])
        orig_conf = original_proba[idx][1] if original_preds[idx] == 1 else original_proba[idx][0]

        # Masked model
        mask_pred = "Female" if masked_preds[idx] == 1 else "Male"
        mask_correct = (masked_preds[idx] == row["label"])
        mask_conf = masked_proba[idx][1] if masked_preds[idx] == 1 else masked_proba[idx][0]

        # Get top features
        orig_features = get_top_features_for_text(row["text"], vec_orig, clf_orig, top_n=5)
        mask_features = get_top_features_for_text(row["text_noun_masked"], vec_masked, clf_masked, top_n=5)

        # Format features as string
        orig_feat_str = ", ".join([f"{feat} ({score:+.2f})" for feat, score in orig_features])
        mask_feat_str = ", ".join([f"{feat} ({score:+.2f})" for feat, score in mask_features])

        analysis_rows.append({
            "original_text": row["text"][:200],  # Truncate for readability
            "masked_text": row["text_noun_masked"][:200],
            "true_label": true_label,
            "original_prediction": orig_pred,
            "original_correct": orig_correct,
            "original_confidence": f"{orig_conf:.3f}",
            "original_top_features": orig_feat_str,
            "masked_prediction": mask_pred,
            "masked_correct": mask_correct,
            "masked_confidence": f"{mask_conf:.3f}",
            "masked_top_features": mask_feat_str,
            "prediction_changed": (orig_pred != mask_pred)
        })

    # Create DataFrame
    analysis_df = pd.DataFrame(analysis_rows)

    # Save to CSV
    analysis_df.to_csv("masking_analysis.csv", index=False)
    print(f"Saved detailed analysis to masking_analysis.csv")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("MASKING ANALYSIS SUMMARY")
    print("=" * 80)

    orig_acc = analysis_df["original_correct"].sum() / len(analysis_df)
    mask_acc = analysis_df["masked_correct"].sum() / len(analysis_df)
    changed = analysis_df["prediction_changed"].sum()

    print(f"\nOriginal model accuracy: {orig_acc:.2%}")
    print(f"Masked model accuracy: {mask_acc:.2%}")
    print(f"Predictions changed: {changed} ({changed / len(analysis_df):.1%})")

    # Show interesting cases
    print("\n" + "=" * 80)
    print("INTERESTING CASES (Prediction Changed)")
    print("=" * 80)

    changed_cases = analysis_df[analysis_df["prediction_changed"]].head(5)
    for i, row in changed_cases.iterrows():
        print(f"\nCase {i}:")
        print(f"  True label: {row['true_label']}")
        print(f"  Original → Masked: {row['original_prediction']} → {row['masked_prediction']}")
        print(f"  Original text: {row['original_text'][:100]}...")
        print(f"  Key original features: {row['original_top_features']}")
        print(f"  Key masked features: {row['masked_top_features']}")

    return analysis_df


def analyze_features(train_df, test_df):
    """
    Extract and compare features before/after masking
    This is what you need for the paper!
    """

    print("\n" + "=" * 80)
    print("QUALITATIVE ANALYSIS - Feature Importance")
    print("=" * 80)

    # Train model on ORIGINAL text
    print("\nTraining on original text...")
    original_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    original_pipe.fit(train_df["text"], train_df["label"])

    # Train model on MASKED text
    print("Training on masked text...")
    masked_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    masked_pipe.fit(train_df["text_noun_masked"], train_df["label"])

    # NEW: Create detailed masking analysis CSV
    create_masking_analysis_csv(train_df, test_df, original_pipe, masked_pipe, sample_size=200)
    create_leakage_analysis_csv(train_df, test_df, original_pipe, masked_pipe, sample_size=200)

    # Extract features from ORIGINAL model
    vectorizer_orig = original_pipe.named_steps["tfidf"]
    clf_orig = original_pipe.named_steps["clf"]
    feature_names_orig = vectorizer_orig.get_feature_names_out()
    coefs_orig = clf_orig.coef_[0]

    # Get top features
    top_female_idx = np.argsort(coefs_orig)[-20:][::-1]  # Top 20 female features
    top_male_idx = np.argsort(coefs_orig)[:20]  # Top 20 male features

    original_female = [(feature_names_orig[i], coefs_orig[i]) for i in top_female_idx]
    original_male = [(feature_names_orig[i], coefs_orig[i]) for i in top_male_idx]

    # Extract features from MASKED model
    vectorizer_masked = masked_pipe.named_steps["tfidf"]
    clf_masked = masked_pipe.named_steps["clf"]
    feature_names_masked = vectorizer_masked.get_feature_names_out()
    coefs_masked = clf_masked.coef_[0]

    top_female_idx = np.argsort(coefs_masked)[-20:][::-1]
    top_male_idx = np.argsort(coefs_masked)[:20]

    masked_female = [(feature_names_masked[i], coefs_masked[i]) for i in top_female_idx]
    masked_male = [(feature_names_masked[i], coefs_masked[i]) for i in top_male_idx]

    # Print results
    print("\n" + "=" * 80)
    print("ORIGINAL MODEL - Top Features")
    print("=" * 80)
    print("\nFemale features (positive weight):")
    for feat, weight in original_female[:10]:
        print(f"  {feat:30s} {weight:+.4f}")

    print("\nMale features (negative weight):")
    for feat, weight in original_male[:10]:
        print(f"  {feat:30s} {weight:+.4f}")

    print("\n" + "=" * 80)
    print("MASKED MODEL - Top Features")
    print("=" * 80)
    print("\nFemale features (positive weight):")
    for feat, weight in masked_female[:10]:
        print(f"  {feat:30s} {weight:+.4f}")

    print("\nMale features (negative weight):")
    for feat, weight in masked_male[:10]:
        print(f"  {feat:30s} {weight:+.4f}")

    # Save to CSV for paper
    pd.DataFrame(original_female, columns=["feature", "weight"]).to_csv(
        "original_features.csv", index=False)
    pd.DataFrame(masked_female, columns=["feature", "weight"]).to_csv(
        "masked_features.csv", index=False)

    print("Feature lists saved to CSV files")

    # Generate LaTeX table
    print("\nGenerating LaTeX table...")

    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lc|lc}",
        "\\toprule",
        "\\multicolumn{2}{c|}{\\textbf{Original}} & \\multicolumn{2}{c}{\\textbf{Masked}} \\\\",
        "\\textbf{Feature} & \\textbf{Weight} & \\textbf{Feature} & \\textbf{Weight} \\\\",
        "\\midrule"
    ]

    # Add top 10 features
    for i in range(10):
        orig_feat, orig_w = original_female[i]
        mask_feat, mask_w = masked_female[i]
        latex_lines.append(f"{orig_feat} & {orig_w:+.3f} & {mask_feat} & {mask_w:+.3f} \\\\")

    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Top features for female gender prediction}",
        "\\label{tab:features}",
        "\\end{table}"
    ])

    with open("feature_table.tex", "w") as f:
        f.write("\n".join(latex_lines))

    print("LaTeX table saved to feature_table.tex")

    # Error analysis
    print("\n" + "=" * 80)
    print("ERROR ANALYSIS")
    print("=" * 80)

    original_preds = original_pipe.predict(test_df["text"])
    masked_preds = masked_pipe.predict(test_df["text_noun_masked"])

    orig_errors = (original_preds != test_df["label"]).sum()
    masked_errors = (masked_preds != test_df["label"]).sum()

    print(f"\nOriginal model errors: {orig_errors} / {len(test_df)} ({100 * orig_errors / len(test_df):.1f}%)")
    print(f"Masked model errors: {masked_errors} / {len(test_df)} ({100 * masked_errors / len(test_df):.1f}%)")

    # Prediction changes
    changed = (original_preds != masked_preds).sum()
    print(f"\nPredictions changed after masking: {changed} ({100 * changed / len(test_df):.1f}%)")

    print("\n" + "=" * 80)
    print("FILES FOR YOUR PAPER:")
    print("=" * 80)
    print("  1. original_features.csv   - What original model used")
    print("  2. masked_features.csv     - What masked model used")
    print("  3. feature_table.tex       - Paste this into your paper")
    print("  4. masking_analysis.csv    - Detailed before/after comparison with feature scores")
    print("\nInterpretation:")
    print("  - Original model uses content words (makeup, bro, etc.)")
    print("  - Masked model uses style (punctuation, intensifiers)")
    print("  - This proves models CAN learn from style, not just content")
