import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load Pre-trained LLM (Falcon, Mistral, or OpenAI GPT)
MODEL_NAME = "tiiuae/falcon-7b-instruct"  # Choose "mistralai/Mistral-7B-Instruct" if needed

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype=torch.float16
)
qa_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)

def generate_questions(text):
    """Generates multiple questions from a given text."""
    prompt = f"Generate 3 questions based on the following passage:\n{text}\nQuestions:"
    output = qa_pipeline(prompt, max_length=200, num_return_sequences=1)[0]["generated_text"]
    
    # Extract questions from output
    questions = [q.strip() for q in output.split("\n") if q.strip()]
    return questions[:3]  # Limit to 3 questions per passage

def generate_qa_pairs(document_text):
    """Splits the document into sections and generates QA pairs."""
    sections = document_text.split("\n\n")  # Split into paragraphs
    qa_pairs = []

    for section in sections:
        if len(section) < 50:  # Skip very short sections
            continue

        questions = generate_questions(section)
        for question in questions:
            qa_pairs.append({"question": question, "answer": section})

    return qa_pairs

if __name__ == "__main__":
    # Load input document
    with open("output.txt", "r", encoding="utf-8") as file:
        document_text = file.read()

    # Generate QA pairs
    qa_dataset = generate_qa_pairs(document_text)

    # Save dataset
    with open("qa_dataset.json", "w", encoding="utf-8") as file:
        json.dump(qa_dataset, file, indent=4)

    print(f"Generated {len(qa_dataset)} QA pairs and saved to qa_dataset.json")
