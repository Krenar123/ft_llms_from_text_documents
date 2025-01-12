import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the Vicuna model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Create a text generation pipeline using Vicuna
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

# Function to summarize text
def summarize_answer(text):
    prompt = f"Summarize the following text in a concise manner:\n\n{text}\n\nSummary:"
    
    # Generate summary using Vicuna
    summary_output = summarizer(prompt, max_length=300, num_return_sequences=1, temperature=0.4)
    
    # Extract the summary (removing extra text if needed)
    summarized_answer = summary_output[0]['generated_text'].split("Summary:")[-1].strip()
    
    return summarized_answer


# Example input
text_to_summarize = """Article 2: (1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies, and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations.
(4) The University reserves the right to take disciplinary action against students who violate the law or University policies."""

# Run summarization
summary = summarize_answer(text_to_summarize)

print("Summarized Text:\n", summary)
