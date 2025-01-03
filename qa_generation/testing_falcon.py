from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Falcon-Instruct model
model_name = "tiiuae/falcon-7b-instruct"  # Falcon-7B-Instruct
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Specify the device: 'cpu'
device = torch.device("cpu")

# Load the model on the CPU
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Input context
context = """
Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following:
RULE ON STUDENT CONDUCT

I. Purpose
Article 1
This Rule regulates student conduct and disciplinary action. It deals with expected standards of behaviour, and action(s) or behaviour which are unacceptable and which have a real or potentially adverse effect. It includes both informal and formal steps of disciplinary procedure.
"""

# Prompt for QA generation
prompt = (
    "Generate specific questions from the following legal text. Include references to articles or sections in the questions. "
    "Provide the output in this format:\n\nQ: [Generated Question]\nA: [Answer]\n\n"
    f"Text:\n{context}"
)

# Tokenize and generate output
inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)
inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to CPU
outputs = model.generate(
    **inputs,
    max_length=512,  # Generate concise output
    temperature=0.7,  # Adjust creativity
    top_k=50,  # Increase diversity
    top_p=0.9,  # Nucleus sampling
)

# Decode and print generated QA pairs
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated QA Pairs:")
print(generated_text)
