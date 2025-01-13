import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Load the Mistral model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    device_map="auto", 
    load_in_8bit=True  # Use 8-bit for efficiency
)

# Assign pad_token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load the dataset
with open("qa_pairs_after_summarize.json", "r", encoding="utf-8") as file:
    qa_data = json.load(file)

# Convert dataset into formatted prompt-response pairs
def convert_to_prompt_input_output(data):
    formatted_data = []
    for item in data:
        question = item["question"]
        answer = item["answer"]
        summary = item.get("summarize", "")

        prompt = f"Question: {question}\nAnswer:"
        full_answer = f"{answer}\nSummary: {summary}" if summary else answer

        formatted_data.append({"prompt": prompt, "response": full_answer})
    return formatted_data

qa_data_converted = convert_to_prompt_input_output(qa_data)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(qa_data_converted)

# Tokenization function
def tokenize_data(sample):
    inputs = tokenizer(sample["prompt"], truncation=True, padding="max_length", max_length=1024)
    labels = tokenizer(sample["response"], truncation=True, padding="max_length", max_length=1024)

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(tokenize_data)

# LoRA Fine-tuning Configuration
lora_config = LoraConfig(
    r=32,  # Increased from 16 to 32
    lora_alpha=64,  # Increased from 32 to 64
    lora_dropout=0.1,  # Increased dropout for stability
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=1,  # Small batch size to fit VRAM
    gradient_accumulation_steps=32,  # Better stability
    evaluation_strategy="no",
    save_steps=500,
    logging_steps=50,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=1e-6,  # Lower LR
    lr_scheduler_type="cosine",  # Better LR decay
    warmup_ratio=0.1,  # 10% warmup
    weight_decay=0.01,
    max_grad_norm=1.0,  # Clip gradients
    fp16=True,
    push_to_hub=True,
    hub_model_id="krenard/mistral-automated-qapairs-finetuned",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save and push the model
trainer.push_to_hub()
print("Fine-tuned model saved and uploaded to Hugging Face!")
