import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load the input JSON file
input_file = "qa_pairs.json"  # Replace with your actual file name
output_file = "summarized_qa_pairs.json"

# Read the JSON file
with open(input_file, "r", encoding="utf-8") as file:
    data = json.load(file)

# Load Falcon 7B model and tokenizer for summarization
# both lmsys/vicuna-7b-v1.5, tiiuae/falcon-7b-instruct
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="auto")

# Create a summarization pipeline using Falcon 7B model
summarizer = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Function to generate new QA pairs with summaries
def augment_dataset(data):
    new_data = []
    
    for item in data:
        question = item["question"]
        original_answer = item["answer"]

        # Generate summary using Falcon 7B
        summary_input = f"Summarize: {original_answer}"
        summary_output = summarizer(summary_input, max_length=700, min_length=30, num_return_sequences=1)

        # Extract the summarized answer
        summarized_answer = summary_output[0]['generated_text']

        # Keep the original QA pair
        new_data.append({"question": question, "answer": original_answer})

        # Add new QA pair with the summarized answer
        new_data.append({"question": question, "answer": f"Summary: {summarized_answer}"})
    
    return new_data

# Process the dataset
augmented_data = augment_dataset(data)

# Save the new dataset to a JSON file
with open(output_file, "w", encoding="utf-8") as file:
    json.dump(augmented_data, file, indent=2, ensure_ascii=False)

print(f"Generated {len(augmented_data)} QA pairs and saved to {output_file}")
