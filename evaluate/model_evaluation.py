import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # Import PeftModel to apply LoRA adapter
import evaluate  # Import evaluate library
from rouge_score import rouge_scorer

def load_model(model_name, adapter_path=None):
    """Loads a base model and optionally applies a LoRA adapter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )

    # Apply the LoRA adapter if provided
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)

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
    return scores["rouge1"], scores["rougeL"]

def evaluate_models(base_model_name, fine_tuned_adapter_path, test_data):
    """Evaluates the base model vs. the fine-tuned adapter."""
    base_model, base_tokenizer = load_model(base_model_name)
    fine_tuned_model, fine_tuned_tokenizer = load_model(base_model_name, fine_tuned_adapter_path)
    
    for sample in test_data:
        question, expected = sample["question"], sample["expected_answer"]
        
        base_response = generate_response(base_model, base_tokenizer, question)
        fine_tuned_response = generate_response(fine_tuned_model, fine_tuned_tokenizer, question)
        
        base_ppl = compute_perplexity(base_model, base_tokenizer, question)
        fine_tuned_ppl = compute_perplexity(fine_tuned_model, fine_tuned_tokenizer, question)
        
        #base_f1 = compute_f1(base_response, base_tokenizer, expected)
        #fine_tuned_f1 = compute_f1(fine_tuned_response, fine_tuned_tokenizer, expected)
        
        base_em = compute_exact_match(base_response, expected)
        fine_tuned_em = compute_exact_match(fine_tuned_response, expected)
        
        base_rouge1, base_rougeL = compute_rouge(base_response, expected)
        fine_tuned_rouge1, fine_tuned_rougeL = compute_rouge(fine_tuned_response, expected)
        
        print(f"Question: {question}")
        print(f"Base Model: {base_response}")
        print(f"Fine-Tuned Model: {fine_tuned_response}")
        print(f"Expected Answer: {expected}")
        print(f"Perplexity - Base: {base_ppl:.2f}, Fine-Tuned: {fine_tuned_ppl:.2f}")
        #print(f"F1 Score - Base: {base_f1:.2f}, Fine-Tuned: {fine_tuned_f1:.2f}")
        print(f"Exact Match - Base: {base_em}, Fine-Tuned: {fine_tuned_em}")
        print(f"ROUGE-1 - Base: {base_rouge1:.2f}, Fine-Tuned: {fine_tuned_rouge1:.2f}")
        print(f"ROUGE-L - Base: {base_rougeL:.2f}, Fine-Tuned: {fine_tuned_rougeL:.2f}")
        print("-" * 80)

if __name__ == "__main__":
    base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"  # Base model
    fine_tuned_adapter_path = "krenard/mistral-automated-qapairs-finetuned"  # LoRA adapter path

    test_data = [
        {"question": "What are the standards of student behavior?", "expected_answer": "(1) Students are members of society and the academic community with attendant rights and responsibilities.  (2) Students are expected to comply with the general law, University policies and campus regulations.  (3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations.  Students who fail to conduct themselves in such a manner may be disciplined. "},
        {"question": "What happens if a student breaks the rules?", "expected_answer": "Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations.  Students who fail to conduct themselves in such a manner may be disciplined."},
    ]
    
    evaluate_models(base_model_name, fine_tuned_adapter_path, test_data)
