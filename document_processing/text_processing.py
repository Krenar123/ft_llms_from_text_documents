import re

def clean_text(text):
    # removing extra blank lines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    
    text = re.sub(r"[^\x00-\x7F]+", "", text)  # removing non-ASCII characters

    return text
