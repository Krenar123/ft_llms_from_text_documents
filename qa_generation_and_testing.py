import torch
import json
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from datasets import Dataset

# Load text from output.txt instead of processing a PDF
with open("output.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)

docs_processed = text_splitter.split_text(text)

# Load Hugging Face Mistral model
# lmsys/vicuna-7b-v1.5
# mistralai/Mistral-7B-Instruct-v0.3
model_name = "AMead10/Llama-3.2-3B-Instruct-AWQ"  # Change to preferred model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# Function to generate QA pairs
def qa_generator_llm(context: str):
    generation_prompt = f"""
    Your task is to write a factoid question and an answer given a context.
    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
    Do NOT include phrases like "according to the passage" or "context".

    Provide your answer as follows:

    Output:::
    Factoid question: (your factoid question)
    Answer: (your answer to the factoid question)

    Now here is the context.

    Context: {context}
    Output:::
    """

    inputs = tokenizer(generation_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=700, temperature=0.5, top_p=0.99)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    try:
        question = generated_text.split("Factoid question: ")[-1].split("Answer: ")[0].strip()
        answer = generated_text.split("Answer: ")[-1].strip()
        return question, answer
    except:
        return None, None

# Generate QA pairs
outputs = []
for doc in tqdm(docs_processed):
    question, answer = qa_generator_llm(doc)
    if question and answer and len(answer) < 700:
        outputs.append({
            "context": doc,
            "question": question,
            "answer": answer,
        })

# Function to evaluate QA pairs
def judge_llm(context: str, question: str, answer: str):
    critique_prompt = f"""
You will be given a question, answer, and a context.
Your task is to provide a total rating using the additive point scoring system described below.
Points start at 0 and are accumulated based on the satisfaction of each evaluation criterion:

Evaluation Criteria:
- Groundedness: Can the question be answered from the given context? (+1)
- Stand-alone: Is the question understandable without the context? (+1)
- Faithfulness: Is the answer grounded in the context? (+1)
- Answer Relevance: Does the answer actually address the question? (+1)

Provide your answer as follows:

Answer:::
Evaluation: (your reasoning)
Total rating: (0 to 4)

Now here are the question, answer, and context.

Question: {question}
Answer: {answer}
Context: {context}
Answer::: """

    inputs = tokenizer(critique_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_length=800, temperature=0.1, top_p=0.99)

    evaluation_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    try:
        score = int(evaluation_text.split("Total rating: ")[-1].strip())
        eval_text = evaluation_text.split("Total rating: ")[-2].split("Evaluation: ")[1].strip()
        return score, eval_text
    except:
        return 0, "Parsing error"

# Evaluate each QA pair
for output in tqdm(outputs):
    score, eval_text = judge_llm(output["context"], output["question"], output["answer"])
    output.update({"score": score, "eval": eval_text})

# Filter high-quality QA pairs
final_dataset = [doc for doc in outputs if doc["score"] >= 4]

# Save to DataFrame
df = pd.DataFrame(final_dataset)
df.to_csv("filtered_qa_dataset.csv", index=False)

 # Save as JSON
with open("./qa_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, ensure_ascii=False, indent=4)
print("Dataset saved as JSON: qa_dataset.json")

# Convert to Hugging Face dataset and save
#dataset = Dataset.from_pandas(df)
#dataset.save_to_disk("./qa_dataset")

print("Processing complete. High-quality QA pairs saved.")
