from prepare_data import load_data, split_by_author, remove_near_duplicates
from masking import apply_masking
from train_models import run_experiments
from qualitative import analyze_features


def main():
    """Run the complete pipeline"""

    print("\n" + "=" * 80)
    print("REDDIT GENDER CLASSIFICATION PIPELINE")
    print("=" * 80)

    # Step 1: Load and prepare data
    print("\n### STEP 1: LOAD DATA ###")
    df = load_data()
    train_df, test_df = split_by_author(df)
    test_df = remove_near_duplicates(train_df, test_df)

    # Step 2: Apply masking
    print("\n### STEP 2: APPLY MASKING ###")
    train_df, test_df = apply_masking(train_df, test_df)

    # Step 3: Train and evaluate models
    print("\n### STEP 3: TRAIN MODELS ###")
    results_df = run_experiments(train_df, test_df)

    # Step 4: Qualitative analysis (Bilgin's part)
    print("\n### STEP 4: QUALITATIVE ANALYSIS ###")
    analyze_features(train_df, test_df)

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - results.csv              (quantitative results)")
    print("  - original_features.csv    (features before masking)")
    print("  - masked_features.csv      (features after masking)")
    print("  - feature_table.tex        (LaTeX table for paper)")


if __name__ == "__main__":
    main()