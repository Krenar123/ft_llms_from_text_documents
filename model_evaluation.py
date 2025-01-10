import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # Import evaluate library
from rouge_score import rouge_scorer

def load_model(model_name):
    """Loads a model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    """Generates a response using the model."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda")
    
    output_ids = model.generate(input_ids, max_length=300)
    return tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

def compute_perplexity(model, tokenizer, text):
    """Computes perplexity of the model on a given text."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def compute_f1(prediction, reference):
    """Computes F1 score for generated text."""
    f1_metric = evaluate.load("f1")  # Using evaluate for F1
    return f1_metric.compute(predictions=[prediction], references=[reference])["f1"]

def compute_exact_match(prediction, reference):
    """Computes exact match score (1 if identical, 0 otherwise)."""
    return int(prediction.strip().lower() == reference.strip().lower())

def compute_rouge(prediction, reference):
    """Computes ROUGE-1 and ROUGE-L scores."""
    rouge_metric = evaluate.load("rouge")  # Using evaluate for ROUGE
    scores = rouge_metric.compute(predictions=[prediction], references=[reference])
    return scores["rouge1"].fmeasure, scores["rougeL"].fmeasure

def evaluate_models(model_name, test_data):
    """Evaluates the base model vs. the fine-tuned model."""
    model, tokenizer = load_model(model_name)
    
    for sample in test_data:
        question, expected = sample["question"], sample["expected_answer"]
        
        response = generate_response(model, tokenizer, question)
        
        ppl = compute_perplexity(model, tokenizer, question)
        f1 = compute_f1(response, expected)
        em = compute_exact_match(response, expected)
        rouge1, rougeL = compute_rouge(response, expected)
        
        print(f"Question: {question}")
        print(f"Model Response: {response}")
        print(f"Expected Answer: {expected}")
        print(f"Perplexity: {ppl:.2f}")
        print(f"F1 Score: {f1:.2f}")
        print(f"Exact Match: {em}")
        print(f"ROUGE-1: {rouge1:.2f}")
        print(f"ROUGE-L: {rougeL:.2f}")
        print("-" * 80)

if __name__ == "__main__":
    model_name = "krenard/mistral7b-merged-qapairs"  # Full fine-tuned model

    test_data = [
        {"question": "What are the standards of student behavior?", "expected_answer": "Students must comply with university policies."},
        {"question": "What happens if a student breaks the rules?", "expected_answer": "They may face disciplinary actions."},
    ]
    
    evaluate_models(model_name, test_data)
