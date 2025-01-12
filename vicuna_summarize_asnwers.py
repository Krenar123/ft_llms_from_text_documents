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
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# Function to summarize each answer in the dataset
def summarize_answer(example):
    original_answer = example["answer"]
    prompt = f"Summarize the following text in a concise manner:\n\n{original_answer}\n\nSummary:"
    
    # Extract the summary (removing extra text if needed)
    summarized_answer = summary_output[0]['generated_text'].split("Summary:")[-1].strip()
    
    # Generate summary using Vicuna model
    summary_output = summarizer(prompt, max_length=700, num_return_sequences=1, temperature=0.7)
    
    # Extract the summarized answer
    summarized_answer = summary_output[0]['generated_text'].strip()
    
    # Return both original and summarized answers in the example
    return {
        "question": example["question"],
        "answer": example["answer"],
        "summarized_answer": summarized_answer
    }

# Apply the summarization function to the entire dataset
summarized_dataset = qa_dataset.map(summarize_answer, batched=False)

# Save the summarized data to a new JSON file
summarized_data = summarized_dataset.to_pandas().to_dict(orient="records")
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(summarized_data, file, indent=2, ensure_ascii=False)

print(f"Generated summarized QA pairs and saved to {output_file}")
