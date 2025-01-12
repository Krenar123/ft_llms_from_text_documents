import json

# Load your dataset from JSON file
with open("qa_pairs_after_summarize.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # Assuming it's a list of QA pairs

# Convert to instruction-response format
jsonl_data = []
for item in data:
    formatted_entry = {
        "instruction": item["question"],  # Question as instruction
        "input": "",  # No additional input
        "output": f"{item['answer']}\n\nSummary: {item['summarize']}"  # Answer + Summary
    }
    jsonl_data.append(formatted_entry)

# Save as JSONL file
with open("formatted_qa_pairs_after_summarize.jsonl", "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")  # Write each dict as a new line

print("✅ Dataset successfully converted to JSONL format!")
