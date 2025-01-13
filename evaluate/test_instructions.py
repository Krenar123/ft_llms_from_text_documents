from transformers import pipeline

question = "What are the standards of student behavior?"
generator = pipeline("text-generation", model="krenard/mistral-automated-qapairs-finetuned-instructions", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
