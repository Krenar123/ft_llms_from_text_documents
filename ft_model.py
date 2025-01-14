import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Load the Mistral model and tokenizer
MODEL_NAME = "meta-llama/Llama-3.2-1B"
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
with open("qa_pairs.json", "r", encoding="utf-8") as file:
    qa_data = json.load(file)

# Convert dataset into formatted prompt-response pairs
def convert_to_prompt_input_output(data):
    formatted_data = []
    for item in data:
        question = item["question"]
        answer = item["answer"]
        summary = item.get("summarize", "")

        prompt = f"SEEU STUDENT QUESTION: {question}\n"
        full_answer = f"SEEU ADMINISTRATION ANSWER: {answer} \nSummary: {summary}" if summary else answer

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
    r=16,  # Increased from 16 to 32
    lora_alpha=32,  # Increased from 32 to 64
    lora_dropout=0.5,  # Increased dropout for stability
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=2,  # Reduce batch size to avoid memory overflow
    gradient_accumulation_steps=8,  # More steps to smooth updates
    evaluation_strategy="no",
    save_steps=500,
    logging_steps=50,
    save_total_limit=3,
    num_train_epochs=5,  # Increase epochs to allow proper convergence
    learning_rate=5e-6,  # Reduce LR further for stability
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,  # Stabilizes early training
    max_grad_norm=1.0,  # 🔥 **CRITICAL FIX: Clipping prevents gradient explosion**
    weight_decay=0.01,
    fp16=True, #bf16=True if torch.cuda.is_bf16_supported() else False,
    optim="adamw_bnb_8bit",  # Optimized for QLoRA
    push_to_hub=True,  
    hub_model_id="krenard/llama3-2-automated-qapairs-finetuned-duplicates",
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
