from transformers import AutoTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("krenard/llama3-2-automated-qapairs-finetuned")
tokenizer = AutoTokenizer.from_pretrained("krenard/llama3-2-automated-qapairs-finetuned")

prompt = "SEEU STUDENT QUESTION: How does SEEU address non-academic misconduct, such as theft or vandalism?\nSEEU ADMINISTRATION ANSWER:"
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=300)
s = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(s)