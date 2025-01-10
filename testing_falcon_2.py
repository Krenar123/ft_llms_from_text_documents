from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# Load the Vicuna model and tokenizer
model_name = "tiiuae/falcon-7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.float16
)

# Set pad_token_id to ensure padding tokens are handled correctly
tokenizer.pad_token = tokenizer.eos_token  # This can be adjusted depending on the model

def chunk_text(text, max_length, overlap=50):
    """Split the text into manageable chunks with overlap."""
    words = text.split()
    for i in range(0, len(words), max_length - overlap):
        yield " ".join(words[i:i + max_length])

def generate_qa_pairs(chunk):
    """Generate as many meaningful question-answer pairs as possible from a text chunk."""
    prompt = (
        "From the following text, generate as many meaningful question-answer pairs as possible. "
        "Include references to articles, sections, or context where applicable. "
        "Output the pairs in this format:\n"
        "[{'question': '...', 'answer': '...'}, {'question': '...', 'answer': '...'}]\n\n"
        f"Text:\n{chunk}\n\nQA Pairs:"
    )
    # Tokenize and generate the attention mask
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    
    # Ensure attention_mask is set and pad_token_id is used
    inputs = {key: value.to("cuda") for key, value in inputs.items()}  # Move tensors to GPU
    
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # Add attention mask here
        pad_token_id=tokenizer.pad_token_id,  # Specify pad token ID
        max_length=1024,
        num_beams=5,
        temperature=0.7,
        no_repeat_ngram_size=2
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def process_document(file_path):
    """Read a text document, chunk it, and generate QA pairs."""
    with open(file_path, "r") as file:
        text = file.read()

    max_chunk_words = 1000  # Define chunk size
    overlap = 50  # Add overlap to maintain context
    qa_pairs = []

    for chunk in chunk_text(text, max_chunk_words, overlap):
        chunk_qa = generate_qa_pairs(chunk)
        try:
            # Parse the generated text into JSON-like structure
            qa_pairs.extend(json.loads(chunk_qa))
        except json.JSONDecodeError:
            print(f"Failed to parse QA pairs for chunk: {chunk[:100]}...")
    return qa_pairs

# File path to your text document
file_path = "output.txt"

# Generate QA pairs
qa_pairs = process_document(file_path)

# Save the results to a file
with open("qa_pairs_2.json", "w") as output_file:
    json.dump(qa_pairs, output_file, indent=4)

# Print generated QA pairs
for qa in qa_pairs:
    print(f"Q: {qa['question']}\nA: {qa['answer']}\n")
