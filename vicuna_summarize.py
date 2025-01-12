import json
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Load Vicuna model and tokenizer
model_name = "lmsys/vicuna-7b-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Create a text generation pipeline using Vicuna
summarizer = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=250)

# Function to summarize an answer
def summarize_answer(answer):
    prompt = f"Summarize the following text in a concise manner based on question and context:\n Question:{question}\n\Context:{answer}\n\nSummary:"
    
    # Generate summary using Vicuna
    summary_output = summarizer(prompt, max_length=1000, num_return_sequences=1, temperature=0.4)
    
    # Extract the summary text
    summarized_text = summary_output[0]['generated_text'].split("Summary:")[-1].strip()
    
    return summarized_text

# Load the input JSON file
#input_file = "qa_pairs.json"  # Update with your actual file name
#output_file = "summarized_qa_pairs.json"

# Read the JSON file
#with open(input_file, "r", encoding="utf-8") as file:
#    data = json.load(file)

# Process each QA pair and add a summary
#processed_data = []
#for item in data:
#    question = item["question"]
#    answer = item["answer"]
#    
    # Generate summary
#    summarized_answer = summarize_answer(answer)
    
    # Store new QA pair with summary
#    processed_data.append({
#        "question": question,
#        "answer": answer,
#        "summarize": summarized_answer
#    })

# Save the new dataset to a JSON file
#with open(output_file, "w", encoding="utf-8") as file:
#    json.dump(processed_data, file, indent=2, ensure_ascii=False)

question = "How does the University expect students to uphold the principle of honesty in their academic work?"
answer = "The principle of honesty must be upheld if the integrity of scholarship is to be maintained by an academic community. The University expects that students will honour this principle and in so doing protect the validity of University learning and academic standards. This means that all academic work will be done by the student to whom it is assigned, without unauthorized aid of any kind."
a = summarize_answer(question, answer)
print(a)
#print(f"Summarization complete! Saved to {output_file}")
