"""
spaCy-based feature extraction for stylometric analysis
"""
import spacy
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter
import gc


def load_spacy_model():
    """Load spaCy model with optimizations"""
    # Disable unnecessary components for speed
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return nlp


def extract_spacy_features(texts, batch_size=100, chunk_size=2000):
    """
    Extract stylometric features using spaCy for linguistic analysis
    
    Features extracted:
    - POS tag distributions
    - Dependency parse patterns
    - Sentence structure metrics
    - Punctuation patterns
    - Function word usage
    
    Args:
        texts: List of text strings
        batch_size: Batch size for spaCy processing
        chunk_size: Process in chunks to manage memory
    
    Returns:
        DataFrame with extracted features
    """
    print(f"\nExtracting spaCy linguistic features from {len(texts)} texts...")
    print("This uses spaCy for POS tagging, dependency parsing, and syntactic analysis")
    
    nlp = load_spacy_model()
    
    all_features = []
    num_chunks = (len(texts) - 1) // chunk_size + 1
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(texts))
        chunk_texts = texts[start_idx:end_idx]
        
        print(f"  Processing chunk {chunk_idx + 1}/{num_chunks} ({len(chunk_texts)} texts)...")
        
        chunk_features = []
        
        # Process with spaCy
        for doc in tqdm(nlp.pipe(chunk_texts, batch_size=batch_size), 
                       total=len(chunk_texts),
                       desc=f"  Chunk {chunk_idx + 1}",
                       leave=False):
            features = _extract_doc_features(doc)
            chunk_features.append(features)
        
        all_features.extend(chunk_features)
        
        # Memory management
        del chunk_features
        gc.collect()
        
        print(f"  ✓ Chunk {chunk_idx + 1}/{num_chunks} complete")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    print(f"\n✓ Extracted {len(df.columns)} spaCy-based features")
    print(f"  Feature categories: POS tags, dependencies, punctuation, syntax")
    
    return df


def _extract_doc_features(doc):
    """Extract features from a single spaCy Doc object"""
    
    # Basic counts
    num_tokens = len(doc)
    num_words = sum(1 for token in doc if not token.is_punct and not token.is_space)
    num_sentences = len(list(doc.sents))
    
    # Avoid division by zero
    if num_tokens == 0:
        return _empty_features()
    
    # POS tag distribution
    pos_counts = Counter(token.pos_ for token in doc)
    pos_features = {
        'pos_noun_ratio': pos_counts.get('NOUN', 0) / num_tokens,
        'pos_verb_ratio': pos_counts.get('VERB', 0) / num_tokens,
        'pos_adj_ratio': pos_counts.get('ADJ', 0) / num_tokens,
        'pos_adv_ratio': pos_counts.get('ADV', 0) / num_tokens,
        'pos_pron_ratio': pos_counts.get('PRON', 0) / num_tokens,
        'pos_det_ratio': pos_counts.get('DET', 0) / num_tokens,
        'pos_adp_ratio': pos_counts.get('ADP', 0) / num_tokens,  # Prepositions
        'pos_conj_ratio': pos_counts.get('CCONJ', 0) / num_tokens,
        'pos_intj_ratio': pos_counts.get('INTJ', 0) / num_tokens,  # Interjections
        'pos_propn_ratio': pos_counts.get('PROPN', 0) / num_tokens,
    }
    
    # Dependency parse patterns
    dep_counts = Counter(token.dep_ for token in doc)
    dep_features = {
        'dep_nsubj_ratio': dep_counts.get('nsubj', 0) / num_tokens,
        'dep_dobj_ratio': dep_counts.get('dobj', 0) / num_tokens,
        'dep_aux_ratio': dep_counts.get('aux', 0) / num_tokens,
        'dep_neg_ratio': dep_counts.get('neg', 0) / num_tokens,
        'dep_advmod_ratio': dep_counts.get('advmod', 0) / num_tokens,
        'dep_amod_ratio': dep_counts.get('amod', 0) / num_tokens,
    }
    
    # Punctuation patterns
    punct_counts = Counter(token.text for token in doc if token.is_punct)
    punct_features = {
        'punct_period_ratio': punct_counts.get('.', 0) / num_tokens,
        'punct_comma_ratio': punct_counts.get(',', 0) / num_tokens,
        'punct_exclaim_ratio': punct_counts.get('!', 0) / num_tokens,
        'punct_question_ratio': punct_counts.get('?', 0) / num_tokens,
        'punct_ellipsis_ratio': punct_counts.get('...', 0) / num_tokens,
        'punct_total_ratio': sum(1 for t in doc if t.is_punct) / num_tokens,
    }
    
    # Sentence structure
    if num_sentences > 0:
        avg_sent_length = num_tokens / num_sentences
        sentence_lengths = [len([t for t in sent]) for sent in doc.sents]
        sent_length_std = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
    else:
        avg_sent_length = 0
        sent_length_std = 0
    
    structure_features = {
        'avg_sentence_length': avg_sent_length,
        'sentence_length_std': sent_length_std,
        'num_sentences': num_sentences,
    }
    
    # Token-level features
    token_features = {
        'avg_token_length': np.mean([len(token.text) for token in doc if not token.is_space]) if num_tokens > 0 else 0,
        'stopword_ratio': sum(1 for t in doc if t.is_stop) / num_tokens,
        'uppercase_ratio': sum(1 for t in doc if t.text.isupper() and len(t.text) > 1) / num_tokens,
    }
    
    # Combine all features
    all_features = {
        **pos_features,
        **dep_features,
        **punct_features,
        **structure_features,
        **token_features
    }
    
    return all_features


def _empty_features():
    """Return zero features for empty documents"""
    return {
        # POS features
        'pos_noun_ratio': 0, 'pos_verb_ratio': 0, 'pos_adj_ratio': 0,
        'pos_adv_ratio': 0, 'pos_pron_ratio': 0, 'pos_det_ratio': 0,
        'pos_adp_ratio': 0, 'pos_conj_ratio': 0, 'pos_intj_ratio': 0,
        'pos_propn_ratio': 0,
        # Dependency features
        'dep_nsubj_ratio': 0, 'dep_dobj_ratio': 0, 'dep_aux_ratio': 0,
        'dep_neg_ratio': 0, 'dep_advmod_ratio': 0, 'dep_amod_ratio': 0,
        # Punctuation features
        'punct_period_ratio': 0, 'punct_comma_ratio': 0, 'punct_exclaim_ratio': 0,
        'punct_question_ratio': 0, 'punct_ellipsis_ratio': 0, 'punct_total_ratio': 0,
        # Structure features
        'avg_sentence_length': 0, 'sentence_length_std': 0, 'num_sentences': 0,
        # Token features
        'avg_token_length': 0, 'stopword_ratio': 0, 'uppercase_ratio': 0,
    }


def extract_pos_ngrams(texts, n=2, max_ngrams=100, batch_size=100):
    """
    Extract POS tag n-grams as features (e.g., NOUN-VERB, ADJ-NOUN)
    These capture syntactic patterns independent of lexical content
    
    Args:
        texts: List of text strings
        n: N-gram size (default 2 for bigrams)
        max_ngrams: Maximum number of n-grams to keep
        batch_size: Batch size for spaCy processing
    
    Returns:
        DataFrame with POS n-gram features
    """
    print(f"\nExtracting POS {n}-grams using spaCy...")
    
    nlp = load_spacy_model()
    
    # Collect all POS n-grams
    all_ngrams = []
    
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size), total=len(texts), desc="Extracting POS patterns"):
        pos_sequence = [token.pos_ for token in doc if not token.is_space]
        
        # Extract n-grams
        doc_ngrams = []
        for i in range(len(pos_sequence) - n + 1):
            ngram = '-'.join(pos_sequence[i:i+n])
            doc_ngrams.append(ngram)
        
        all_ngrams.append(Counter(doc_ngrams))
    
    # Get most common n-grams across all documents
    total_counts = Counter()
    for ngram_counter in all_ngrams:
        total_counts.update(ngram_counter)
    
    top_ngrams = [ngram for ngram, _ in total_counts.most_common(max_ngrams)]
    
    # Create feature matrix
    features = []
    for ngram_counter in all_ngrams:
        doc_features = {f'pos_{ngram}': ngram_counter.get(ngram, 0) for ngram in top_ngrams}
        features.append(doc_features)
    
    df = pd.DataFrame(features).fillna(0)
    
    print(f"✓ Extracted {len(df.columns)} POS {n}-gram features")
    
    return df
