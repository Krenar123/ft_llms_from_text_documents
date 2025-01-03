from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import json
from datasets import load_dataset


def fine_tune_model(data_path):
  # Here we need to change the data path to notebooklm_qa_pairs.json
  dataset = load_dataset("json", data_files={"train": data_path})

  model_name = "mistralai/Mistral-7B-Instruct-v0.3"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      load_in_8bit=True, # for memory optimization
      device_map="auto"
  )


  def preprocess_function(example):
      prompt = f"Question: {example['question']}\nAnswer:"
      labels = example['answer']
      tokenized = tokenizer(prompt, truncation=True, padding="max_length", max_length=512)
      labels_tokenized = tokenizer(labels, truncation=True, padding="max_length", max_length=512)
      tokenized["labels"] = labels_tokenized["input_ids"]
      return tokenized

  tokenized_dataset = dataset["train"].map(preprocess_function, batched=True)


  training_args = TrainingArguments(
      output_dir="./mistral_qa_pairs_t5", # here we need to change the name based on qa dataset used e.g _notebooklm, _openai
      evaluation_strategy="steps",
      save_strategy="steps",
      per_device_train_batch_size=1, # since we have 2 gpus we might want to change this to 2. I will leave it as 1 as not sure what errors this might give
      gradient_accumulation_steps=8, # effective larger batch size, if we have 2 as btach size we can do 2 on this one as well it will be enough
      num_train_epochs=3,
      save_total_limit=2,
      logging_dir="./logs",
      learning_rate=5e-5,
      fp16=True, # add this below ddp_find_unused_parameters=False, only when used 2 gpus
  )

  trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_dataset,
      tokenizer=tokenizer,
  )

  # train the model
  trainer.train()
