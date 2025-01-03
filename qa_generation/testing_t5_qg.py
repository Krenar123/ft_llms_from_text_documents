from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("valhalla/t5-small-e2e-qg")
model = AutoModelForSeq2SeqLM.from_pretrained("valhalla/t5-small-e2e-qg")

# Define a function for generating questions
def generate_questions(text):
    # Add the "generate questions:" prefix as required by the model
    input_text = f"generate question: {text}"
    
    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True)
    
    # Generate questions
    outputs = model.generate(inputs, max_length=256, num_return_sequences=1, num_beams=1, early_stopping=True)
    
    questions = set()  # Use a set to remove duplicates
    for output in outputs:
        result = tokenizer.decode(output, skip_special_tokens=True)
        questions.add(result.strip())
  
    # Remove empty strings and strip whitespace
    return questions

# Example text
txt = "Based on Article 94, paragraph 1, item 43 of the Law on Higher Education (Official Gazette of Republic of Macedonia, number 82/2018 and Official Gazette of Republic of North Macedonia, number 154/2019, 76/2020 and 178/2021), and Article 31, paragraph 1, item 30 Statute of South East European University, the University Senate, at its meeting held on 28.09.2021 approved the following: RULE ON STUDENT CONDUCT"
text = """
Principles of Academic Integrity
"""

# Generate unique questions
questions = generate_questions(text)
print(questions)
