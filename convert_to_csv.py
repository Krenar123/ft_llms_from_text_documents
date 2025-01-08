import json
import pandas as pd

# Load the JSON data from qa_pairs.json file
with open('converted_qa_pairs_notebooklm.json', 'r') as file:
    data = json.load(file)

# Flatten the 'messages' into the required format
rows = []
for conversation in data['messages']:
    # Create tuple of message pairs (user, system)
    rows.append({"messages": conversation})

# Create a DataFrame with each conversation as one row
df = pd.DataFrame(rows)

# Save the DataFrame to a CSV file
df.to_csv('notebooklm_messages.csv', index=False)

# Print the DataFrame to check the result
print(df)
