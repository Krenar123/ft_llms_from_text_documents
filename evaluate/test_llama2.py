import torch
from transformers import pipeline

model_id = "krenard/llama3-2-automated-qapairs-finetuned-duplicates"
# krenard/llama3-2-automated-qapairs-finetuned
pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

# Specify the max_length parameter
output = pipe("SEEU STUDENT QUESTION: What measures does the Security Service take to ensure the safety and well-being of residents in dormitories/housing facilities?\nSEEU ADMINISTRATION ANSWER:", max_length=500)  # Adjust max_length as needed

print(output)
