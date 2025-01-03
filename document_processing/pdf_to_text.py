import fitz  # PyMuPDF for PDF processing
import re

def pdf_to_text(pdf_path):
    doc = fitz.open(pdf_path)
    structured_text = ""
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")

        if "CONTENTS" in text or "Contents" in text:
            continue  # skipping this page or section entirely(contents page)
        
        # check the text to find sections, subsections, etc.
        for line in text.splitlines():
            stripped_line = line.strip()
            if re.match(r"^(.*?)(\.+)\s*(\d+)$", stripped_line) and stripped_line.count('.') > 3:
                continue  # skipping if it has contents
            
            if re.match(r"^\d+\.\s", line):
                structured_text += f"\n\n# {line}\n"
            elif re.match(r"^\d+\.\d+\s", line):
                structured_text += f"\n## {line}\n"
            elif re.match(r"^\d+\.\d+\.\d+\s", line):
                structured_text += f"\n### {line}\n"
            else:
                structured_text += line + "\n"
    
    doc.close()
    print(structured_text[:len(structured_text)//4])
    return structured_text