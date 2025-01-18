import json

# Load your dataset from JSON file
with open("qa_pairs_after_summarize.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # Assuming it's a list of QA pairs

# Convert to instruction-response format
jsonl_data = []
for item in data:
    formatted_entry = {
        "instruction": f"SEEU student question: {item['question']}",  # Question as instruction
        "input": "",  # No additional input
        "output": f"{item['summarize']}\n\n{item['answer']}"  # Answer + Summary
    }
    jsonl_data.append(formatted_entry)

# Save as JSONL file
with open("instructions_formatted_qa_pairs_with_summary.jsonl", "w", encoding="utf-8") as f:
    for entry in jsonl_data:
        f.write(json.dumps(entry) + "\n")  # Write each dict as a new line

print("✅ Dataset successfully converted to JSONL format!")
