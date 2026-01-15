from prepare_data import load_data, split_by_author, remove_near_duplicates
from masking import apply_masking
from train_models import run_experiments, run_spacy_experiments
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

    # Step 2: Apply noun removal using spaCy
    print("\n### STEP 2: NOUN REMOVAL (SPACY) ###")
    train_df, test_df = apply_masking(train_df, test_df)

    # Step 3: Train and evaluate models
    print("\n### STEP 3: TRAIN MODELS ###")
    results_df = run_experiments(train_df, test_df)

    # Step 4: Qualitative analysis (Bilgin's part)
    print("\n### STEP 4: QUALITATIVE ANALYSIS ###")
    analyze_features(train_df, test_df)

    # Step 5: spaCy-based experiments (OPTIONAL - takes longer)
    print("\n### STEP 5: SPACY EXPERIMENTS (OPTIONAL) ###")
    print("This step uses spaCy for linguistic feature extraction")
    print("It will take longer but demonstrates pure stylometric analysis")

    run_spacy = input("\nRun spaCy experiments? (y/n): ").strip().lower() == 'y'

    if run_spacy:
        spacy_results = run_spacy_experiments(train_df, test_df)
    else:
        print("Skipping spaCy experiments")

    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print("  - results.csv              (quantitative results: original vs noun-removed)")
    print("  - original_features.csv    (features before noun removal)")
    print("  - masked_features.csv      (features after noun removal)")
    print("  - feature_table.tex        (LaTeX table for paper)")
    print("  - masking_analysis.csv     (detailed before/after comparison with feature scores)")
    if run_spacy:
        print("  - spacy_results.csv        (results using spaCy linguistic features)")
        print("  - train_spacy_features.csv (extracted spaCy features for training)")
        print("  - test_spacy_features.csv  (extracted spaCy features for testing)")
    print("\nNote: All noun removal uses spaCy for POS tagging")


if __name__ == "__main__":
    main()