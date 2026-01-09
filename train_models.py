import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def run_experiments(train_df, test_df):
    """Run all experiments - same as notebook"""

    # Define models
    models = {
        "MultinomialNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "LinearSVM": LinearSVC()
    }

    # Define settings
    settings = {
        "original": "text",
        "leakage_masked": "text_leakage_masked",
        "noun_masked": "text_noun_masked"
    }

    # Run experiments
    results = []
    print("\nTraining models...")

    for model_name, model in models.items():
        print(f"\n{model_name}:")
        for setting_name, column in settings.items():
            # Create pipeline
            pipe = Pipeline([
                ("tfidf", TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=5,
                    max_df=0.9,
                    stop_words="english"
                )),
                ("clf", model)
            ])

            # Train and evaluate
            pipe.fit(train_df[column], train_df["label"])
            preds = pipe.predict(test_df[column])

            acc = accuracy_score(test_df["label"], preds)
            f1 = f1_score(test_df["label"], preds, average="macro")

            results.append({
                "model": model_name,
                "setting": setting_name,
                "accuracy": acc,
                "macro_f1": f1
            })

            print(f"  {setting_name:20s} Acc: {acc:.4f}, F1: {f1:.4f}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print("\nMacro-F1:")
    print(results_df.pivot(index="model", columns="setting", values="macro_f1"))

    print("\nAccuracy:")
    print(results_df.pivot(index="model", columns="setting", values="accuracy"))

    # Save
    results_df.to_csv("results.csv", index=False)
    print("\nResults saved to results.csv")

    return results_df