import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate  # Import evaluate library

def load_model(model_name):
    """Loads a model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer

def format_chat_template(messages):
    """Formats messages into a string for chat-based models."""
    formatted_input = ""
    for message in messages:
        if message["role"] == "user":
            formatted_input += f"User: {message['content']}\n"
        elif message["role"] == "assistant":
            formatted_input += f"Assistant: {message['content']}\n"
    return formatted_input.strip()

def generate_response(model, tokenizer, prompt):
    """Generates a response using the model."""
    # If chat-like input is required, format it
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = format_chat_template(messages)

    # Tokenize input and generate response
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to("cuda")
    output_ids = model.generate(input_ids, max_length=300)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def compute_perplexity(model, tokenizer, text):
    """Computes perplexity of the model on a given text."""
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return torch.exp(loss).item()

def compute_bleu(prediction, reference):
    """Computes BLEU score for generated text."""
    bleu_metric = evaluate.load("bleu")  # Using evaluate for BLEU
    return bleu_metric.compute(predictions=[prediction], references=[[reference]])["bleu"]

def compute_exact_match(prediction, reference):
    """Computes exact match score (1 if identical, 0 otherwise)."""
    return int(prediction.strip().lower() == reference.strip().lower())

def compute_rouge(prediction, reference):
    """Computes ROUGE-1 and ROUGE-L scores."""
    rouge_metric = evaluate.load("rouge")  # Using evaluate for ROUGE
    scores = rouge_metric.compute(predictions=[prediction], references=[reference])
    return scores["rouge1"], scores["rougeL"]

def evaluate_models(base_model_name, fine_tuned_model_name, test_data):
    """Evaluates the base model vs. the fine-tuned model."""

    # Load both models
    base_model, base_tokenizer = load_model(base_model_name)
    fine_tuned_model, fine_tuned_tokenizer = load_model(fine_tuned_model_name)

    for sample in test_data:
        question, expected = sample["question"], sample["expected_answer"]

        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, question)
        fine_tuned_response = generate_response(fine_tuned_model, fine_tuned_tokenizer, question)

        # Compute perplexity
        base_ppl = compute_perplexity(base_model, base_tokenizer, question)
        fine_tuned_ppl = compute_perplexity(fine_tuned_model, fine_tuned_tokenizer, question)

        base_bleu = compute_bleu(base_response, expected)
        fine_tuned_bleu = compute_bleu(fine_tuned_response, expected)

        base_em = compute_exact_match(base_response, expected)
        fine_tuned_em = compute_exact_match(fine_tuned_response, expected)

        base_rouge1, base_rougeL = compute_rouge(base_response, expected)
        fine_tuned_rouge1, fine_tuned_rougeL = compute_rouge(fine_tuned_response, expected)

        # Print results
        print(f"Question: {question}")
        print(f"Expected Answer: {expected}")
        print("-" * 80)
        print(f"Base Model ({base_model_name}):")
        print(f"Response: {base_response}")
        print(f"Perplexity: {base_ppl:.2f}")
        print(f"BLEU Score: {base_bleu:.2f}")
        print(f"Exact Match: {base_em}")
        print(f"ROUGE-1: {base_rouge1:.2f}, ROUGE-L: {base_rougeL:.2f}")
        print("-" * 80)
        print(f"Fine-Tuned Model ({fine_tuned_model_name}):")
        print(f"Response: {fine_tuned_response}")
        print(f"Perplexity: {fine_tuned_ppl:.2f}")
        print(f"BLEU Score: {fine_tuned_bleu:.2f}")
        print(f"Exact Match: {fine_tuned_em}")
        print(f"ROUGE-1: {fine_tuned_rouge1:.2f}, ROUGE-L: {fine_tuned_rougeL:.2f}")
        print("=" * 100)

if __name__ == "__main__":
    base_model_name = "meta-llama/Llama-3.2-1B"  # Base model
    fine_tuned_model_name = "krenard/llama3-2-automated-qualitative-qapairs-finetuned-instructions"  # Fine-tuned full model
    test_data = [
        {
            "question": "SEEU student question: What are the standards of student behavior?",
            "expected_answer": "(1) Students are members of society and the academic community with attendant rights and responsibilities. (2) Students are expected to comply with the general law, University policies and campus regulations. (3) Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined."
        },
        {
            "question": "SEEU student question: What happens if a student breaks the rules?",
            "expected_answer": "Students on University property or attending any official University function assume an obligation to conduct themselves in a manner compatible with University policies and campus rules and regulations. Students who fail to conduct themselves in such a manner may be disciplined."
        },
        {
            "question": "SEEU student question: Who will be involved in the decision-making process and how long will it take?",
            "expected_answer": "With regard to more general misconduct, violations or attempted violations include, but are not limited to: Knowingly giving or disseminating false information; Failure to comply with a reasonable instruction, including refusing to identify oneself, given by a university official, on university property or at any external, officially organized event; or obstructing officials in carrying out their duty; Forgery, alteration, or misuse of any University document, record, key, electronic device or identification; Unauthorized entry to, possession of, receipt of, duplication of, or use of the campus or Universitys name, insignia, or seal; Theft, damage or destruction of University property or the property of others while on University premises; Misuse of computing facilities or computer time, as described in the Rule on Computer and Network Use; Unauthorized entry to, possession of, receipt, or use of any campus or University property, equipment, resources, or services; Violation of policies, regulations, or rules governing campus or University- owned or operated housing facilities or leased housing facilities; Knowingly reporting a false emergency; Verbal abuse, or more sustained harassment, either directed at an individual or group, including derogatory references to race, ethnicity, religion, gender, sexual orientation, disability, or other personal/group characteristics; Physical abuse such as physical assault, threats of violence, or conduct that threatens the health or safety of any person; Disorderly or lewd conduct such as swearing and drunkenness; Unlawful or attempted manufacture, distribution, dispensing, possession, use, or sale of narcotics or other illegal substances; or unauthorized sale or use of alcohol; Actual or attempted manufacture, possession, storage, or use of fireworks, explosives and/or explosive devices, firearms or other dangerous or destructive devices or weapons."
        },
        {
            "question": "SEEU student question: How does the University Disciplinary Commission decide on cases of major disciplinary offences, and what are the possible outcomes?",
            "expected_answer": "The University Disciplinary Commission investigates and makes decisions or proposals about cases of alleged major disciplinary offences, both academic and general, established by the Rector.\n\nThe University Disciplinary Commission, established by the Rector, investigates and makes decisions or proposals about cases pertaining to alleged major disciplinary offences, both academic and general."
        },
        {
            "question": "SEEU student question: Who is involved in the decision-making process for dormitory-related incidents, and how long does it take to reach a decision?",
            "expected_answer": "The Head, another Security Operator, and/or the Manager of dormitories/housing facilities will be involved in the decision-making process. The decision will be made within one week after the evidence is collected.\n\nThe Head, with another Security Operator and/or the Manager of dormitories/housing facilities meet to analyse the evidence and make a decision within one week after the evidence is collected."
        },
    ]
    
    evaluate_models(base_model_name, fine_tuned_model_name, test_data)
