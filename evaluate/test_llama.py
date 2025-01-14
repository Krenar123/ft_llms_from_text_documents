import torch
from transformers import pipeline

model_id = "krenard/llama3-2-automated-qapairs-finetuned-instructions"

pipe = pipeline(
    "text-generation", 
    model=model_id, 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

pipe("How does SEEU address non-academic misconduct, such as theft or vandalism?")