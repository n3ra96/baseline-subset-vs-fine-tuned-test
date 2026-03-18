# Baseline: Prompt-Based Text Classification

## Overview
This module implements a **zero-shot text classification baseline** using a pretrained instruction-tuned LLM (`FLAN-T5`).

Instead of training, the model is prompted to classify each input into one of the predefined labels.

---

## Approach

- Use an instruction prompt:
- Explicitly list all possible labels
- Force the model to return **only one label**
- Perform inference using `generate()`
- Normalize outputs to match valid labels
- Evaluate predictions using standard classification metrics

---

## Pipeline

1. Load dataset (`data`, `labels`)
2. Build prompt template
3. Run inference per sample
4. Normalize predictions
5. Evaluate:
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix

---

## Key Characteristics

### Advantages
- No training required
- Fast to prototype
- Flexible for new tasks

### Limitations
- Sensitive to prompt wording
- Output may require parsing
- Lower accuracy compared to fine-tuned models

---

## Example Prompt
  Input text:
  The government announced new policies to improve economic growth.
  
  Model prediction:
  business
  
  True label:
  business
  
---

## Output

- Classification report (precision / recall / F1)
- Confusion matrix
- Sample predictions

---

## When to Use

This approach is useful when:
- You need a quick baseline
- No labeled training data is available
- You want to test LLM capabilities without training

---

## Model Used

- `google/flan-t5-base`

---

## Summary

This baseline serves as a reference point to evaluate how much improvement is gained through fine-tuning.
