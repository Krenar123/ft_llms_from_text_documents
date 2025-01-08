import json

# Load the original QA pairs from the JSON file
with open("notebooklm_qa_pairs.json", "r") as f:
    qa_data = json.load(f)

# Convert the QA pairs to the desired format
converted_data = []
for item in qa_data:
    # Convert each question-answer pair into the desired format
    converted_data.append({"content": item["question"], "role": "user"})
    converted_data.append({"content": item["answer"], "role": "system"})

# Wrap the converted data in a list of lists as per your required format
wrapped_data = []
for i in range(0, len(converted_data), 2):
    wrapped_data.append([converted_data[i], converted_data[i + 1]])

# Print the converted data as JSON
converted_json = json.dumps(wrapped_data, indent=4)
print(converted_json)

# Optionally save to a new file
with open("converted_qa_pairs_notebooklm.json", "w") as f:
    json.dump(wrapped_data, f, indent=4)
