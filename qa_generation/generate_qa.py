import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the Flan-T5-large model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

# Load the input JSON file
input_file = "./outputs/processed_documents/rule_on_student_conduct_sep_2021_v2.json"
with open(input_file, "r") as file:
    data = json.load(file)

# Extract the main_title from the first element
first_key = next(iter(data))  # Get the first key in the JSON
main_title = data[first_key].get("main_title", "")  # Extract the main_title

# Output JSON structure
output_qa_pairs = []

# Function to generate questions based on the section name
def generate_questions(section_name, section_text):
    main_prompt = (
        "You are an AI assistant trained to generate meaningful and diverse questions. "
        "Based on the following section title and its purpose, generate specific, highly relevant questions. "
        "Avoid general or repetitive questions.\n"
        f"Section Title: {section_name}\nContent: {section_text}"
    )
    
    prompt = (
        "You are an AI assistant trained to generate meaningful and diverse questions. "
        "Based on the following section title and its purpose, generate specific, highly relevant questions. "
        "Avoid general or repetitive questions.\n"
        f"Section Title: {section_name}"
    )

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    # Generate the output
    outputs = model.generate(
        **inputs,
        max_length=156,  # Ensure concise output
        num_return_sequences=3,  # Generate multiple outputs
        num_beams=3,  # Reduce the beams for better diversity
        temperature=0.4,  # Add randomness for variety
        do_sample=True,
        early_stopping=True
    )

    # Decode, clean, and filter results
    questions = set()  # Use a set to remove duplicates
    for output in outputs:
        result = tokenizer.decode(output, skip_special_tokens=True)
        questions.add(result.strip())

    return list(questions)

# Additional feature: Split article text into parts and generate QA pairs
def process_numbered_parts(article_text):
    # Split based on numbered patterns like (1), (2), etc.
    parts = re.split(r"\(\d+\)", article_text)
    # Re-add the split marker for each part
    numbered_parts = [f"({i + 1}) {part.strip()}" for i, part in enumerate(parts) if part.strip()]
    
    part_qa_pairs = []  # List to store QA pairs for parts

    for part in numbered_parts:
        # Generate a question for each part
        prompt_text = (
            f"You are an AI assistant trained to generate questions. "
            f"Based on the following content, generate a highly relevant question.\n"
            f"Content: {part}"
        )
        
        question = generate_questions_from_text(prompt_text)  # Generate a single question
        part_qa_pairs.append({"question": question, "answer": part})  # Add the QA pair
    
    return part_qa_pairs

# Function to generate a single question from text
def generate_questions_from_text(prompt_text):
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length=156,
        num_return_sequences=3,
        num_beams=3,
        temperature=0.3,
        do_sample=True,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Iterate through the JSON structure and generate QA pairs
for section_name, section_content in data.items():
    if "articles" in section_content:
        # Combine all articles' texts into a single string for the answer
        combined_text = " ".join(
            [f"{article['title']}: {article['article_text']}" for article in section_content["articles"]]
        )
        
        # Generate questions based on the section name
        questions = generate_questions(section_name, combined_text)

        # Add each QA pair to the output structure
        for question in questions:
            output_qa_pairs.append({"question": question, "answer": combined_text})

        # Add section-level QA pair: section name as the question, combined article texts as the answer
        output_qa_pairs.append({
            "question": section_name,
            "answer": combined_text
        })

        # Add article-level QA pairs: section name + article title as the question, article text as the answer
        for article in section_content["articles"]:
            output_qa_pairs.append({
                "question": f"{section_name} - {article['title']}",
                "answer": article["article_text"]
            })

        # Add section-level QA pair: main_title + section_name as the question, combined article texts as the answer
        output_qa_pairs.append({
            "question": f"{main_title} - {section_name}",
            "answer": combined_text
        })

        # Add article-level QA pairs: main_title + section_name + article title as the question, article text as the answer
        for article in section_content["articles"]:
            output_qa_pairs.append({
                "question": f"{main_title} - {section_name} - {article['title']}",
                "answer": article["article_text"]
            })

            # Process numbered parts in article text for additional QA pairs
            numbered_part_pairs = process_numbered_parts(article["article_text"])
            output_qa_pairs.extend(numbered_part_pairs)

# Add introduction-level QA pair (if applicable)
if first_key in data:
    intro = data[first_key]
    if "title" in intro and "article_text" in intro:
        output_qa_pairs.append({
            "question": f"{main_title} - {intro['title']}",
            "answer": intro["article_text"]
        })

# Save the output to a JSON file
output_file = "qa_pairs.json"
with open(output_file, "w") as file:
    json.dump(output_qa_pairs, file, indent=4)

print(f"Generated QA pairs saved to {output_file}")
