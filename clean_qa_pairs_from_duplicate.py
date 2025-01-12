import json

# Load the QA pairs from the JSON file
with open('qa_pairs.json', 'r') as file:
    qa_pairs = json.load(file)

# Create a dictionary to store unique questions and their corresponding answers
unique_qa_pairs = {}

for qa_pair in qa_pairs:
    question = qa_pair['question']
    answer = qa_pair['answer']
    
    # Add the question-answer pair only if the question is not already in the dictionary
    if question not in unique_qa_pairs:
        unique_qa_pairs[question] = answer

# Convert the dictionary back into a list of QA pairs
cleaned_qa_pairs = [{"question": question, "answer": answer} for question, answer in unique_qa_pairs.items()]

# Save the cleaned QA pairs back into the JSON file
with open('cleaned_qa_pairs.json', 'w') as file:
    json.dump(cleaned_qa_pairs, file, indent=4)

print("Duplicate QA pairs removed and saved to 'cleaned_qa_pairs.json'")
