# Fine-Tuning Mistral-7B with LoRA

This repository contains scripts to fine-tune **Mistral-7B-Instruct-v0.3** using **LoRA (Low-Rank Adaptation)** for **question-answering (QA) tasks**.

## Install Dependencies

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo.git
   cd your-repo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Fine-Tuning Script

Before fine-tuning you have to login to your own huggingface account:

```bash
huggingface-cli login
```

To fine-tune the model, run:

```bash
python main.py
```

Worth noting that you must have access to https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3


## Multi-GPU Training (Distributed Data Parallel)

If using **two GPUs**, modify `fine_tuning/fine_tuning_llms.py` by adding the following changes in `TrainingArguments`:

```python
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=2, # increase batch size for multiple GPUs
    gradient_accumulation_steps=4, # change it for effective batch size
    ddp_find_unused_parameters=False,  # enable the Distributed Data Parallel (DDP)
)
```

Launch training with:

```bash
torchrun --nproc_per_node=2 main.py
```

## Notes

- This script uses **LoRA** to reduce VRAM usage and allow fine-tuning on consumer GPUs.
- If you encounter CUDA memory issues, try reducing `per_device_train_batch_size` or using **`load_in_8bit=True`**.


