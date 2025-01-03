from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the LLaMA-2 model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

# Input context
context = """
Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following:
RULE ON STUDENT CONDUCT

I. Purpose
Article 1
This Rule regulates student conduct and disciplinary action. It deals with expected standards of behaviour, and action(s) or behaviour which are unacceptable and which have a real or potentially adverse effect. It includes both informal and formal steps of disciplinary procedure.

II. Standards of student behaviour
Article 2
(1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined.
"""

# Refined prompt for generating specific questions
prompt = (
    "You are an AI assistant trained to generate meaningful and specific questions. "
    "Generate specific questions based on the following content:\n\n"
    f"{context}"
)

# Tokenize the input
inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

# Generate the output
outputs = model.generate(
    **inputs,
    max_length=150,  # Max length for the generated response
    num_return_sequences=5,  # Generate multiple outputs
    num_beams=10,  # Increase diversity
    early_stopping=True
)

# Decode and print the results
for i, output in enumerate(outputs):
    question = tokenizer.decode(output, skip_special_tokens=True)
    print(f"Question {i+1}: {question.strip()}")
