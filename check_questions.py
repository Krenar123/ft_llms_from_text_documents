from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Load the pre-trained Mistral model and tokenizer
model_name = 'mistralai/Mistral-7B-Instruct-v0.3'  # Mistral model on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # ✅ Fix: Set EOS token as padding
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the JSON file containing the pairs (question, context)
with open('qa_pairs.json', 'r') as f:
    data = json.load(f)

# Function to check if the question makes sense based on context using Mistral
def check_question_with_context(question, context):
    # Combine the context and question as input
    input_text = f"Context: {context}\nQuestion: {question}\nDoes this question make sense based on the context above? Answer with 'Yes' or 'No'."

    # Tokenize the input
    inputs = tokenizer(
        input_text, 
        return_tensors='pt', 
        truncation=True, 
        padding=True,  
        max_length=1024
    )

    # Generate model output
    with torch.no_grad():
        output = model.generate(**inputs, max_length=1024, num_return_sequences=1, do_sample=False)

    # Decode the generated output and extract the answer
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Check if the model's response contains 'Yes' or 'No'
    return "Yes" if "Yes" in generated_text else "No"

valid_qa_pairs = []

# Process each pair and filter those that make sense
for pair in data:
    question = pair['question']
    context = pair['answer']
    
    if check_question_with_context(question, context) == "Yes":
        valid_qa_pairs.append(pair)
        
with open('valid_qa_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(valid_qa_pairs, f, ensure_ascii=False, indent=4)

print(f"✅ Saved {len(valid_qa_pairs)} valid QA pairs to 'valid_qa_pairs.json'")
