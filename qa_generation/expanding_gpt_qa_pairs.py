import json

# Filepath for the current expanded QA set
qa_filepath = "/mnt/data/qa_pairs_expanded.json"

# Load the current QA set
with open(qa_filepath, "r") as file:
    qa_pairs = json.load(file)

# Function to further expand QA pairs from existing data
def expand_qa_pairs(qa_pairs):
    expanded_pairs = []
    
    for pair in qa_pairs:
        question = pair["question"]
        answer = pair["answer"]
        
        # Generate additional questions by rephrasing and focusing on different aspects of the answer
        expanded_pairs.append({"question": question, "answer": answer})  # Original pair
        
        # Rephrase questions for different emphasis
        expanded_pairs.append({
            "question": f"What is meant by '{question}'?",
            "answer": answer
        })
        
        expanded_pairs.append({
            "question": f"Can you explain {question.lower()}?",
            "answer": answer
        })
        
        # If the answer is detailed, create subsets of questions
        if len(answer.split()) > 15:
            # Break down into shorter questions if possible
            parts = answer.split(". ")  # Split by sentences
            for i, part in enumerate(parts):
                if part.strip():
                    expanded_pairs.append({
                        "question": f"Detail part {i+1} of the answer to: {question}",
                        "answer": part.strip() + ("." if not part.strip().endswith(".") else "")
                    })
        
        # Context-specific questions based on phrases in the answer
        if "student" in answer.lower():
            expanded_pairs.append({
                "question": f"What does this rule state about students in the context of '{question}'?",
                "answer": answer
            })
        
        if "disciplinary" in answer.lower():
            expanded_pairs.append({
                "question": f"What disciplinary measures are discussed in '{question}'?",
                "answer": answer
            })
        
        if "university" in answer.lower():
            expanded_pairs.append({
                "question": f"How does the university address the issue in '{question}'?",
                "answer": answer
            })
    
    return expanded_pairs

# Expand the QA pairs
further_expanded_qa_pairs = expand_qa_pairs(qa_pairs)

# Save the further expanded QA set
expanded_filepath = "/mnt/data/qa_pairs_further_expanded.json"
with open(expanded_filepath, "w") as file:
    json.dump(further_expanded_qa_pairs, file, indent=4)

expanded_filepath