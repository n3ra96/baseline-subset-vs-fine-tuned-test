
# Fine-Tuning: LoRA on F2LLM-0.6B

## Overview
This module implements **parameter-efficient fine-tuning (LoRA)** on a small language model (`F2LLM-0.6B`) for text classification.

Instead of prompting the model, we train it directly on labeled data to specialize it for the task.

---

## Approach

- Convert labels → integer IDs
- Use `AutoModelForSequenceClassification`
- Attach **LoRA adapters** to attention layers
- Train only a small subset of parameters

---

## Why LoRA?

LoRA (Low-Rank Adaptation) allows:
- Training only ~0.7% of parameters
- Faster training
- Lower GPU memory usage
- Lightweight checkpoints

---

## Pipeline

1. Load dataset
2. Stratified split:
   - Train / Validation / Test
3. Tokenize text
4. Load base model
5. Add LoRA adapters
6. Train using Hugging Face Trainer
7. Evaluate on test set

---

## Key Design Decisions

### 1. Sequence Classification Head
- Added automatically by Transformers
- Initialized randomly (`score.weight`)
- Learned during fine-tuning

### 2. Padding Token Fix
- Set `pad_token = eos_token`
- Required for batch training

### 3. Mixed Precision (bf16)
- Used instead of fp16 for stability
- Faster and avoids gradient scaling issues

### 4. Gradient Checkpointing
- Reduces memory usage
- Prevents CUDA out-of-memory errors

---

## Training Configuration

- Model: `codefuse-ai/F2LLM-0.6B`
- Method: LoRA
- Batch size: effective ~16
- Sequence length: 192
- Precision: bf16
- Optimizer: fused AdamW

---

## Evaluation Metrics

- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro + weighted)
- Confusion matrix

---

## Output

- Trained LoRA adapters (lightweight)
- Metrics (JSON)
- Predictions (CSV)

---

## Advantages

- Higher accuracy than prompting
- Stable outputs (no parsing needed)
- Efficient training with small GPU

---

## Limitations

- Requires labeled data
- Training time needed
- Less flexible than prompting for new tasks

---

## Summary

Fine-tuning aligns the model with the dataset, leading to significantly improved performance compared to the prompt-based baseline.
