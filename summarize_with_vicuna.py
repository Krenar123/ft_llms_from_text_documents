import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load the Vicuna model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Create a text generation pipeline using Vicuna
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=450)

# Function to summarize text
def summarize_answer(text):
    prompt = f"Your task is to write a factoid question and an answer given a context.\nYour factoid question should be answerable with a specific, concise piece of factual information from the context(write the summary and the actual rules).\n\nContext: {text}\n\nProvide your answer as follows:\nOutput:::\nFactoid question:\n Answer:\nNow here is the context."

    
    # Generate summary using Vicuna
    summary_output = summarizer(prompt, max_length=2000, num_return_sequences=1, temperature=0.7)
    
    # Extract the summary (removing extra text if needed)
    summarized_answer = summary_output[0]['generated_text']
    
    return summarized_answer


# Example input
text_to_summarize = """
RULE ON STUDENT CONDUCT - II. Standards of student behaviour
Article 2: (1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies, and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations.
(4) The University reserves the right to take disciplinary action against students who violate the law or University policies."""

# Run summarization
summary = summarize_answer(text_to_summarize)

print(summary)
