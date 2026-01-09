import re
import spacy
from tqdm import tqdm
from config import *
import gc

# Expand gender terms for regex matching
GENDER_TERMS_EXPANDED = set()
for term in GENDER_TERMS:
    GENDER_TERMS_EXPANDED.add(term)
    GENDER_TERMS_EXPANDED.add(term.lower())
    GENDER_TERMS_EXPANDED.add(term.upper())
    GENDER_TERMS_EXPANDED.add(term.capitalize())

# Pre-compile regex
gender_pattern = re.compile(r'\b(' + '|'.join(re.escape(term) for term in GENDER_TERMS_EXPANDED) + r')\b',
                            re.IGNORECASE)


def mask_leakage_only_regex(texts):
    """Ultra-fast gender masking using regex"""
    out = []
    for text in tqdm(texts, desc="Leakage masking (REGEX)"):
        masked = gender_pattern.sub(LEAKAGE_TOKEN, text)
        out.append(masked)
    return out


def mask_all_nouns_chunked(texts, chunk_size=2000):
    """
    Noun masking in small chunks to avoid RAM issues
    Processes 2000 posts at a time, then clears memory
    """
    print(f"Processing {len(texts)} posts in chunks of {chunk_size} (RAM-friendly)")

    # Load minimal spaCy
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
            tokens = [CONTENT_TOKEN if token.pos_ in ["NOUN", "PROPN"]
                      else token.text for token in doc]
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
    """Apply hybrid masking (regex + chunked)"""
    print("\nApplying masking strategies (HYBRID MODE - RAM-FRIENDLY)...")
    print("- Gender terms: Regex (instant)")
    print("- Nouns: Small chunks to avoid RAM issues")

    # Gender masking with regex (fast!)
    print("\n[1/4] Masking gender terms in train set...")
    train_df["text_leakage_masked"] = mask_leakage_only_regex(train_df["text"].tolist())

    print("\n[2/4] Masking gender terms in test set...")
    test_df["text_leakage_masked"] = mask_leakage_only_regex(test_df["text"].tolist())

    # Noun masking in chunks (RAM-friendly)
    print("\n[3/4] Masking nouns in train set...")
    train_df["text_noun_masked"] = mask_all_nouns_chunked(train_df["text"].tolist(), chunk_size=2000)

    print("\n[4/4] Masking nouns in test set...")
    test_df["text_noun_masked"] = mask_all_nouns_chunked(test_df["text"].tolist(), chunk_size=2000)

    print("\n✓ Masking complete!")

    return train_df, test_df