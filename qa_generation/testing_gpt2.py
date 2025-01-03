from transformers import GPT2LMHeadModel, GPT2Tokenizer
import textwrap

# Load GPT-2 model and tokenizer
model_name = "openai-community/gpt2-large"  # Official GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Input context
txt = "Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following: RULE ON STUDENT CONDUCT"

context = """
Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following:
RULE ON STUDENT CONDUCT

I. Purpose
Article 1
This Rule regulates student conduct and disciplinary action. It deals with expected standards of behaviour, and action(s) or behaviour which are unacceptable and which have a real or potentially adverse effect. It includes both informal and formal steps of disciplinary procedure.

II. Standards of student behaviour
(1) Students are members of society and the academic community with attendant rights and responsibilities.
(2) Students are expected to comply with the general law, University policies and campus regulations.
(3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined.
"""

# Refined prompt with examples
example_qa = textwrap.dedent("""
    You are an AI asistant that analyzes the text below amd generates a question 
    Text:
""")
prompt = example_qa + txt

# Tokenize input
inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True)

# Generate output
outputs = model.generate(
    inputs,
    max_length=512,  # Generate concise output
    num_return_sequences=1,  # One sequence at a time
    temperature=0.7,  # Balance creativity
    no_repeat_ngram_size=2,  # Avoid repetition
    do_sample=True,  # Enable sampling
)

# Decode and print output
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Generated QA Pairs:")
print(generated_text)
