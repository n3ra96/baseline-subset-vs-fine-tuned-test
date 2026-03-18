# baseline-subset-vs-fine-tuned-test
## 1) Local dataset loading (CSV upload → DataFrame)

Change: Load the dataset from an uploaded CSV in Colab and keep it in a pandas df.
Why: Makes the notebook dataset-agnostic and allows running on private/local datasets without Hugging Face Hub.

## 2) Stratified train/val/test split (80/10/10)

Change: Use train_test_split(..., stratify=labels) twice to create train/val/test.
Why: Preserves label distribution across splits; avoids misleading metrics from imbalanced subsets and enables proper “best checkpoint” selection using a validation set.

## 3) Prompt baseline (zero-shot label generation)

Change: Implement a prompt template that lists allowed labels and forces “return only label”.
Why: Creates a true “no training” baseline to compare against fine-tuning. Also demonstrates prompt-engineering + robust parsing.

## 4) Stratified evaluation subset for baseline

Change: Evaluate the prompt baseline on a small stratified subset (e.g., 300 samples).
Why: LLM-style inference is slower/costly. A stratified subset keeps metrics stable while making iteration fast.

## 5) Switch fine-tuning to SequenceClassification (not generative labels)

Change: Fine-tune using AutoModelForSequenceClassification with integer label_id.
Why: F2LLM-0.6B is better suited to embedding/feature-extraction/classification style training; classification head provides stable logits + clean metric computation.

## 6) Padding token fix for batching

Change: Define tokenizer.pad_token (fallback to eos_token) and set model.config.pad_token_id.
Why: Many decoder-style tokenizers ship without pad tokens; batching requires padding. Without this, training fails for batch_size > 1.

## 7) LoRA adapters (PEFT)

Change: Use PEFT LoRA (r=16, alpha=32, dropout=0.05) targeting attention projection modules.
Why: Trains ~0.7% of parameters (few million) instead of full model weights → faster, cheaper, smaller checkpoints, lower VRAM.

## 8) bf16 training on Tesla T4

Change: Use bf16=True and fp16=False.
Why: Avoids FP16 GradScaler errors and is fast/stable on supported GPUs; improves throughput vs pure FP32.

## 9) Memory safeguards

Change: Use gradient_checkpointing_enable() and limit max_length (e.g., 192).
Why: Attention memory scales ~O(L²). Checkpointing and smaller sequence length prevent CUDA OOM while keeping training feasible on a single GPU.

## 10) Reproducible artifacts

Change: Save metrics JSON, predictions CSV, and LoRA adapters.
Why: Makes results reproducible and easy to compare across runs; adapters can be versioned and reloaded without saving full model weights. 
