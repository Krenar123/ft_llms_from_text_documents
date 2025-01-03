# main.py

from document_processing.pdf_to_text import pdf_to_text
from document_processing.text_processing import clean_text
from document_processing.parse_structure import parse_text_into_structure_with_intro, save_to_json
#from qa_generation.generate_qa import generate_qa
#from qa_generation.qa_formatting import save_as_json
#from fine_tuning.ft_model import fine_tune_model
from fine_tuning.fine_tuning_llms import fine_tune_model

def main(pdf_path):
    # 1 convert PDF to structured text
    #print("Converting PDF to structured text...")
    #structured_text = pdf_to_text(pdf_path)
    
    # 2 clean the extracted text
    #print("Cleaning extracted text...")
    #cleaned_text = clean_text(structured_text)
    
    #with open("output.txt", 'w', encoding='utf-8') as txt_file:
    #    txt_file.write(cleaned_text)

    #parsed_text = parse_text_into_structure_with_intro(cleaned_text)
    # 3 generate Q&A pairs from cleaned text
    #print("Generating Q&A pairs...")
    #qa_pairs = generate_qa(cleaned_text)
    
    # 4 save Q&A pairs to JSON format
    #print(f"Saving Q&A pairs to {output_json_path}...")
    #save_as_json(qa_pairs, output_json_path)
    #print("Process completed successfully.")
    #print("--------------")
    #print(parsed_text)
    #save_to_json(parsed_text, "outputs/processed_documents/rule_on_student_conduct_sep_2021_v2.json")
    
    # 5 fine tune model
    # based on all pairs generated, notebook, chatgpt, openai and qa_pairs(auto generated)
    data_path = "qa_pairs.json"
    fine_tune_model(data_path)



if __name__ == "__main__":
    # Specify paths for PDF input and JSON output
    pdf_path = "data/Rule on student conduct Sep 2021.pdf"               # Replace with the path to your PDF file
    #output_json_path = "outputs/qa_dataset.json"  # Replace with the desired output path for JSON file

    main(pdf_path)
