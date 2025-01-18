import json

# List to hold the filtered data
filtered_data = []

# Open the input file and read it line by line
with open('instructions_formatted_qa_pairs_with_summary.jsonl', 'r') as f:
    for line in f:
        # Parse each line as a JSON object
        entry = json.loads(line)
        
        # Filter out entries where the instruction does not end with a question mark
        if entry['instruction'].strip().endswith('?'):
            filtered_data.append(entry)

# Save the filtered data to a new file
with open('instructions_formatted_qa_pairs_with_summary_only_qa.jsonl', 'w') as f:
    for entry in filtered_data:
        json.dump(entry, f)
        f.write("\n")  # Ensure each JSON object is on a new line

print("Filtered data saved to 'filtered_qa_pairs.jsonl'.")
