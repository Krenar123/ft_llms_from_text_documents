from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def fine_tune_model(data_path):
  dataset = load_dataset("json", data_files={"train": data_path})

  model_name = "mistralai/Mistral-7B-Instruct-v0.3"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=True,
    device_map="auto"
  )

  lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
  )

  model = get_peft_model(model, lora_config)

  def preprocess_function(example):
    prompt = f"Question: {example['question']}\nAnswer:"
    labels = example['answer']
    tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
    labels_tokenized = tokenizer(labels, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = labels_tokenized["input_ids"]
    return tokenized

  tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)

  training_args = TrainingArguments(
    output_dir="./mistral_qa_pairs_t5",
    evaluation_strategy="steps",
    save_strategy="steps",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    learning_rate=5e-5,
    fp16=True,
    optim="paged_adamw_8bit",
    lr_scheduler_type="cosine",
    warmup_steps=100,
    ddp_find_unused_parameters=False
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
  )

  trainer.train()
