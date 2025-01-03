import fitz  # PyMuPDF
import openai
import json
import csv

# Set your OpenAI API key

# Step 1: Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Step 2: Chunk the text
def chunk_text(text, max_length=1000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1  # Include space

        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:  # Add any remaining words as the last chunk
        chunks.append(" ".join(current_chunk))

    return chunks

def generate_qa_pairs(chunk, model="gpt-4"):
    prompt = f"""
You are an AI designed to generate meaningful and specific question-answer pairs based on input text.
Please read the following text and create 5 diverse and relevant QA pairs:

Text:
{chunk}

Output:
1. Q: [question] A: [answer]
2. Q: [question] A: [answer]
3. Q: [question] A: [answer]
4. Q: [question] A: [answer]
5. Q: [question] A: [answer]
"""
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    # Extract and return the generated QA pairs
    return response['choices'][0]['message']['content']

# Step 4: Main process
def process_pdf_to_qa(pdf_path, output_json="qa_dataset.json", output_csv="qa_dataset.csv"):
    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_path)
    print("PDF text extracted.")

    # Chunk the text
    chunks = chunk_text(document_text, max_length=1000)
    print(f"Text chunked into {len(chunks)} parts.")

    # Generate QA pairs for each chunk
    qa_pairs = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1} of {len(chunks)}...")
        qa_pairs.append(generate_qa_pairs(chunk))
    print("QA pairs generated.")

    # Save to JSON
    with open(output_json, "w") as f:
        json.dump(qa_pairs, f, indent=4)
    print(f"QA pairs saved to {output_json}.")

    # Save to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Question", "Answer"])
        for qa_chunk in qa_pairs:
            for line in qa_chunk.split("\n"):
                if line.startswith("Q:") and "A:" in line:
                    question, answer = line.split("A:", 1)
                    writer.writerow([question.strip()[3:], answer.strip()])
    print(f"QA pairs saved to {output_csv}.")

# Step 5: Run the script
if __name__ == "__main__":
    pdf_path = "data/Statute SEEU March 2021 - FINAL June 24 MM.pdf"  # Replace with your PDF file path
    process_pdf_to_qa(pdf_path)
