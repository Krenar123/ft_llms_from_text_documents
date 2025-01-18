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
III. Principles of Academic Integrity
(1) The principle of honesty must be upheld if the integrity of scholarship is to be maintained by an academic community. The University expects that students will honour this principle and in so doing protect the validity of University learning and academic standards. This means that all academic work will be done by the student to whom it is assigned, without unauthorized aid of any kind. (2) Students are expected to complete the course in compliance with the published standards and should not engage in any activity that involves attempting to receive a grade by dishonest means. Instructors, for their part, exercise care in planning and supervising academic work, so that academic effort and integrity is encouraged."""

# Run summarization
summary = summarize_answer(text_to_summarize)

print(summary)
