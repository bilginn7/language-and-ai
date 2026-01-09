# Reddit Gender Classification - Clean Code Version

**This is our notebook code cleaned up into separate Python files.**

## Files (5 total)

```
config.py             - Settings (gender terms, file paths)
prepare_data.py       - Load data, split by author, remove duplicates
masking.py            - Mask gender terms and nouns
train_models.py       - Train 3 models on 3 settings
qualitative.py        - Feature analysis (Bilgin's part)
main.py               - Main script (runs everything)
```

## Setup

```bash
pip install pandas numpy scikit-learn spacy tqdm
python -m spacy download en_core_web_sm
```

## Run

```bash
python run_pipeline.py
```

That's it! Takes 5-10 minutes to run.

## What Each File Does

### config.py
- Gender terms list
- File paths
- Settings

### prepare_data.py
```python
load_data()              # Load gender.csv
split_by_author()        # Split train/test by author
remove_near_duplicates() # Clean test set
```

### masking.py
```python
mask_leakage_only()  # Mask gender terms (he, she, etc.)
mask_all_nouns()     # Mask all nouns
apply_masking()      # Apply both strategies
```

### train_models.py
```python
run_experiments()  # Train NB, LogReg, SVM on original/masked data
```

### qualitative.py
```python
analyze_features()  # Extract top features, generate tables
```

### main.py
```python
main()  # Runs everything in order
```

## Output Files

- `results.csv` - Accuracy and F1 scores
- `original_features.csv` - What original model learned
- `masked_features.csv` - What masked model learned  
- `feature_table.tex` - LaTeX table for paper

## Understanding the Flow

```
1. Load gender.csv
   ↓
2. Split by author (no overlap)
   ↓
3. Remove near-duplicates
   ↓
4. Create masked versions:
   - text_leakage_masked (gender terms → [LEAKAGE])
   - text_noun_masked (all nouns → [CONTENT])
   ↓
5. Train 3 models × 3 settings = 9 experiments
   ↓
6. Extract features and analyze
```

## Help

- **Can't find a function?** Check which file makes sense (data stuff in prepare_data.py, model stuff in train_models.py)
- **Want to change settings?** Edit config.py
- **Need to modify masking?** Edit masking.py