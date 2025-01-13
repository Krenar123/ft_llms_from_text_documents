from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import datasets
import torch

# Load Mistral-7B model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Apply QLoRA (Low-Rank Adapters)
lora_config = LoraConfig(
    r=16,  # Low-rank adaptation
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Load dataset
dataset = datasets.load_dataset("json", data_files={"train": "formatted_qa_pairs_after_summarize.jsonl"}, split="train")

# Define training hyperparameters
training_args = TrainingArguments(
    output_dir="./mistral-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=5,  # More epochs to improve learning
    save_strategy="epoch",
    save_total_limit=2,
    evaluation_strategy="epoch",
    learning_rate=1e-4,  # Lower LR to avoid overfitting
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Gradual LR warmup
    logging_steps=50,
    fp16=True,  # Mixed precision training
    optim="adamw_bnb_8bit",  # Optimized for QLoRA
    logging_dir="./logs",
    report_to="none",
    push_to_hub=True,  
    hub_model_id="krenard/mistral-automated-qapairs-finetuned-instructions",
)

# Fine-tune the model
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="instruction",  # Use 'instruction' as the input field
    dataset_target_field="output"  # Use 'output' as the target field
)

trainer.train()

# Save fine-tuned model
trainer.save_model("./mistral-finetuned")
tokenizer.save_pretrained("./mistral-finetuned")
trainer.push_to_hub()
