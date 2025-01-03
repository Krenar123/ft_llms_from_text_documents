from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Flan-T5-large model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Input context
context = """
Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following:
RULE ON STUDENT CONDUCT
"""
text = """
Principles of Academic Integrity
"""

context_2 = """
(1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined.
"""

# Refined prompt
prompt = (
    "Generate a question based on the text below: \n"
    f"Text:\n{text}"
)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

# Generate the output
outputs = model.generate(
    **inputs,
    max_length=156,  # Ensure concise output
    num_return_sequences=1,  # Generate multiple outputs
    num_beams=1,  # Reduce the beams for better diversity
    temperature=0.1,  # Add randomness for variety
    do_sample=True,
    early_stopping=True
)

# Decode, clean, and filter results
questions = set()  # Use a set to remove duplicates
for output in outputs:
    result = tokenizer.decode(output, skip_special_tokens=True)
    questions.add(result.strip())

# Print the results
for i, question in enumerate(questions):
    print(f"Question {i + 1}:\n{question}\n")
