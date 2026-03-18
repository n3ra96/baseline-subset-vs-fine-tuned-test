# Comparison: Prompting vs Fine-Tuning

## Overview
This module compares two approaches for text classification:

1. **Prompt-Based Classification (Baseline)**
2. **LoRA Fine-Tuned Model (F2LLM-0.6B)**

The goal is to evaluate trade-offs between:
- performance
- efficiency
- flexibility

---

## Compared Methods

### 1. Prompt-Based (FLAN-T5)
- Zero-shot classification
- No training required
- Uses instruction prompts

### 2. Fine-Tuned (F2LLM + LoRA)
- Supervised learning
- Task-specific adaptation
- Trains small subset of parameters

---

## Evaluation Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro + weighted)
- Confusion matrix

---

## Results (Example)

| Model | Method | Accuracy | Macro F1 | Notes |
|------|--------|---------|---------|------|
| FLAN-T5 | Prompting | 0.88 | 0.73 | Zero-shot |
| F2LLM-0.6B | LoRA | 0.98 | 0.98 | Fine-tuned |

---

## Key Insights

### 1. Performance
- Fine-tuning significantly improves F1-score
- Better handling of class boundaries

### 2. Consistency
- Prompting may produce noisy or inconsistent outputs
- Fine-tuned model outputs structured predictions

### 3. Cost
- Prompting: no training cost, higher inference cost
- Fine-tuning: upfront training cost, cheaper inference

### 4. Flexibility
- Prompting adapts easily to new tasks
- Fine-tuning requires retraining

---

## When to Use Each

### Use Prompting if:
- No labeled data
- Rapid prototyping
- Dynamic label sets

### Use Fine-Tuning if:
- You have labeled data
- Need high accuracy
- Production deployment

---

## Conclusion

Fine-tuning with LoRA provides a strong improvement over prompt-based classification, especially for structured tasks with fixed label sets.

However, prompt-based methods remain valuable for quick experimentation and low-resource scenarios.

---

## Summary

This comparison highlights the trade-off between:
- **flexibility (prompting)**  
vs  
- **performance (fine-tuning)**

Both approaches are important tools in modern LLM pipelines.

## Methodology

For detailed design decisions and implementation reasoning, see:
[Methodology](docs/methodology.md)
