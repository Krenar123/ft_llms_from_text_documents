import json
import re

# Load the JSON file
input_file = "qa_pairs_vicuna.json"  # Change to your actual file name
output_file = "processed_qa_data_vicuna.json"

# Read the JSON file
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Function to extract numbered questions and create new pairs
def extract_questions(data):
    new_data = []
    
    for item in data:
        original_question = item["question"]
        answer = item["answer"]  # Preserve the answer exactly as it is
        
        # Check if the question contains numbered items (1., 2., 3., etc.)
        if re.search(r"\n?\d+\.\s", original_question):
            # Extract questions using regex
            extracted_questions = re.findall(r"\n?\d+\.\s(.*?)(?=\n\d+\.|\Z)", original_question, re.DOTALL)

            # Create new question-answer pairs while keeping the answer unchanged
            new_data.extend([{"question": q.strip(), "answer": answer} for q in extracted_questions])
        else:
            # Keep the original question-answer pair if no numbered items are found
            new_data.append({"question": original_question.strip(), "answer": answer})
    
    return new_data

# Process the data
processed_data = extract_questions(data)

# Save the new dataset to a JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(processed_data, file, indent=2, ensure_ascii=False)

print(f"Processed {len(processed_data)} new QA pairs and saved to {output_file}")
