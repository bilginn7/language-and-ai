import spacy
from tqdm import tqdm
import gc


def remove_nouns_chunked(texts, chunk_size=2000):
    """
    Noun removal (deletion) using spaCy POS tagging
    Processes texts in chunks to avoid RAM issues

    Args:
        texts: List of text strings
        chunk_size: Number of texts to process at once (default: 2000)

    Returns:
        List of texts with nouns and proper nouns removed
    """
    print(f"Processing {len(texts)} posts in chunks of {chunk_size} (RAM-friendly)")

    # Load minimal spaCy - only need POS tagger
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])

    out = []
    num_chunks = (len(texts) - 1) // chunk_size + 1

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(texts))
        chunk = texts[start_idx:end_idx]

        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk)} posts)...")

        # Process this chunk with small batches
        chunk_results = []
        for doc in tqdm(nlp.pipe(chunk, batch_size=50),
                        total=len(chunk),
                        desc=f"  Chunk {chunk_idx + 1}",
                        leave=False):
            # Only keep tokens that are NOT nouns/proper nouns
            tokens = [token.text for token in doc
                      if token.pos_ not in ["NOUN", "PROPN"]]
            chunk_results.append(" ".join(tokens))

        # Add to output
        out.extend(chunk_results)

        # Clear memory
        del chunk_results
        gc.collect()

        # Show progress
        print(f"  ✓ Chunk {chunk_idx + 1}/{num_chunks} complete ({len(out)}/{len(texts)} total)")

    return out


def apply_masking(train_df, test_df):
    """
    Apply spaCy-based noun removal to train and test sets

    This function uses spaCy's POS tagger to identify and remove all nouns
    and proper nouns, leaving only function words, verbs, adjectives,
    adverbs, and punctuation.

    Args:
        train_df: Training DataFrame with 'text' column
        test_df: Test DataFrame with 'text' column

    Returns:
        Tuple of (train_df, test_df) with added columns:
        - text_noun_removed: Text with nouns removed via spaCy
        - text_leakage_masked: Copy of original (for compatibility)
    """
    print("\n" + "=" * 80)
    print("NOUN REMOVAL USING SPACY")
    print("=" * 80)
    print("\nUsing spaCy POS tagger to identify and remove:")
    print("  - NOUN (common nouns)")
    print("  - PROPN (proper nouns)")
    print("\nThis will leave:")
    print("  - Function words (pronouns, determiners, prepositions)")
    print("  - Verbs and verb forms")
    print("  - Adjectives and adverbs")
    print("  - Punctuation")

    # Noun removal using spaCy
    print("\n[1/2] REMOVING nouns from training set...")
    train_df["text_noun_removed"] = remove_nouns_chunked(train_df["text"].tolist(), chunk_size=2000)

    print("\n[2/2] REMOVING nouns from test set...")
    test_df["text_noun_removed"] = remove_nouns_chunked(test_df["text"].tolist(), chunk_size=2000)

    # For backward compatibility with existing code that expects these column names
    train_df["text_noun_masked"] = train_df["text_noun_removed"]
    test_df["text_noun_masked"] = test_df["text_noun_removed"]

    train_df["text_leakage_masked"] = train_df["text"]  # No masking, just original
    test_df["text_leakage_masked"] = test_df["text"]  # No masking, just original

    print("\n✓ Noun removal complete!")
    print(f"✓ Created columns: text_noun_removed, text_noun_masked")
    print("=" * 80)

    return train_df, test_df