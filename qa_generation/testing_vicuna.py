from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the Flan-T5-large model and tokenizer
# Load Vicuna model and tokenizer (7B version example)
tokenizer = AutoTokenizer.from_pretrained("AMead10/Llama-3.2-3B-Instruct-AWQ")
model = AutoModelForCausalLM.from_pretrained("AMead10/Llama-3.2-3B-Instruct-AWQ")

# Input context
context = """
Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following:
RULE ON STUDENT CONDUCT
"""

text = """
(2) Students are expected to comply with the general law, University policies and campus regulations.
"""

context_2 = """
(1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined.
"""

# Refined prompt
prompt = (
    "You are an AI assistant trained to generate meaningful and diverse questions. "
    "Analyze the title and text below and generate highly specific, non-repetitive questions covering different details and perspectives. "
    "Avoid general or repetitive questions.\n"
    f"Text:\n{context_2}"
)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

# Generate the output
outputs = model.generate(
    **inputs,
    max_length=156,  # Ensure concise output
    num_return_sequences=5,  # Generate multiple outputs
    num_beams=5,  # Reduce the beams for better diversity
    temperature=0.7,  # Add randomness for variety
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
