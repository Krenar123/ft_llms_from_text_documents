from transformers import pipeline

question = "How does SEEU address non-academic misconduct, such as theft or vandalism?"
generator = pipeline("text-generation", model="krenard/llama3-2-automated-qapairs-finetuned", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
