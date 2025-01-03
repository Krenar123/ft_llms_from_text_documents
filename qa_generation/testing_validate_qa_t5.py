from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

context = "..."
question = "..."
answer = "..."
prompt = f"Validate the following QA pair:\n\nContext: {context}\nQ: {question}\nA: {answer}"

inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
