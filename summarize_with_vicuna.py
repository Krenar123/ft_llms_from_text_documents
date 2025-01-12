import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the input JSON file
input_file = "processed_qa_data_vicuna.json"  # Replace with your actual file name
output_file = "summarized_qa_pairs_vicuna.json"

# Read the JSON file
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Load the Vicuna model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Load your dataset into a Hugging Face Dataset object
qa_dataset = Dataset.from_list(data)

# Create a text generation pipeline using Vicuna
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Function to summarize each answer in the dataset
def summarize_answer(example):
    original_answer = example["answer"]
    prompt = f"Please summarize the following text and give me only the summarization:\n{original_answer}"
    
    # Generate summary using Vicuna model
    summary_output = summarizer(prompt, max_length=1000, num_return_sequences=1, temperature=0.4)
    
    # Extract the summarized answer
    summarized_answer = summary_output[0]['generated_text'].strip()
    
    # Return both original and summarized answers in the example
    return summarized_answer
