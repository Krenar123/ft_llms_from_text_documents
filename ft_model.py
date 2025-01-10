import torch
import json
import transformers
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from huggingface_hub import login

# Hugging Face Authentication (replace with your token)
HUGGINGFACE_TOKEN = ""
login(token=HUGGINGFACE_TOKEN)

# Load Mistral-7B model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)

# Load the QA dataset
with open("qa_dataset.json", "r", encoding="utf-8") as file:
    qa_data = json.load(file)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(qa_data)

# Tokenize dataset
def tokenize_data(sample):
    input_text = f"Question: {sample['question']}\nAnswer:"
    output_text = sample["answer"]
    inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
    labels = tokenizer(output_text, truncation=True, padding="max_length", max_length=512)
    
    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(tokenize_data)

# LoRA Fine-tuning Configuration
lora_config = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)

# Apply LoRA to the model
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="steps",
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,  # Upload to Hugging Face
    hub_model_id="krenard/mistral-automated-qapairs-finetuned",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save the model and push to Hugging Face Hub
trainer.push_to_hub()
print("Fine-tuned model saved and uploaded to Hugging Face!")
