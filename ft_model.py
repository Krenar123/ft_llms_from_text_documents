import torch
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from huggingface_hub import login

# Hugging Face Authentication (Uncomment if needed)
# HUGGINGFACE_TOKEN = "your_huggingface_token"  # Add your token here
# login(token=HUGGINGFACE_TOKEN)

# Load the Mistral-7B model and tokenizer
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

# Assign pad_token if not already defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token

# Load the QA dataset
with open("qa_pairs.json", "r", encoding="utf-8") as file:
    qa_data = json.load(file)

# Convert dataset to prompt-input-output format
def convert_to_prompt_input_output(data):
    converted_data = []
    for item in data:
        question = item['question']
        answer = item['answer']
        converted_data.append({
            "question": question,
            "answer": answer
        })
    return converted_data

qa_data_converted = convert_to_prompt_input_output(qa_data)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(qa_data_converted)

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

# Ensure that all model parameters (including LoRA parameters) are trainable
for param in model.parameters():
    param.requires_grad = True  # Unfreeze all parameters (or selectively freeze/unfreeze layers as needed)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mistral_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    eval_strategy="no",  # No evaluation (adjust as needed)
    save_strategy="epoch",
    logging_steps=10,
    save_total_limit=2,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,  # Upload to Hugging Face
    hub_model_id="krenard/mistral-merged-automated-qapairs-finetuned",  # Model ID on Hugging Face
)

# Trainer initialization
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train the model
trainer.train()

# Save and push to Hugging Face Hub
trainer.push_to_hub()
print("Fine-tuned model saved and uploaded to Hugging Face!")
