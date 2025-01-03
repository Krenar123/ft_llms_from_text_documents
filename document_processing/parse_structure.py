import re
import json

def parse_text_into_structure(text):
    # pattern for matching the roman numerals or section titles, articles, and article content
    section_pattern = r"^[IVXLCDM]+\.\s+[A-Za-z0-9\s]+"  # matching Roman numeral sections (e.g., I., II., III.)
    article_pattern = r"^Article \d+"  # matching articles: "Article 1", "Article 2", ...
    
    structure = {"Introduction": {"title": ""}}
    current_section = None
    current_article = None
    article_text = ""
    pre_title = None
    article_title = ""
    introduction_text = ""

    is_intro_section = True

    for line in text.splitlines():
        line = line.strip()

        # if we are checking the intro section (before the first section), add lines to introduction_text
        if is_intro_section:
            if re.match(section_pattern, line):
                structure["Introduction"]["title"] = introduction_text.strip()
                structure["Introduction"]["main_title"] = introduction_text.split(":")[-1].strip()
                is_intro_section = False
                current_section = line
                structure[current_section] = {"articles": []}
                article_text = ""
                pre_title = None
                article_title = ""
            else:
                introduction_text += line + " "
        
        elif re.match(section_pattern, line):
            if current_section:
                if current_article:
                    if article_title.strip() and article_text.strip():
                        structure[current_section]["articles"].append({
                            'pre_title': pre_title if pre_title else "",
                            'title': article_title if article_title else "",
                            'article_text': article_text.strip()
                        })
            
            # Start new section
            current_section = line
            structure[current_section] = {"articles": []}
            article_text = ""
            pre_title = None
            article_title = ""
        
        elif re.match(article_pattern, line):
            if current_article:
                if article_title.strip() and article_text.strip():
                    structure[current_section]["articles"].append({
                        'pre_title': pre_title if pre_title else "",
                        'title': article_title if article_title else "",
                        'article_text': article_text.strip()
                    })
            
            current_article = line
            pre_title = None
            article_title = line.strip()
            article_text = ""
            
        else:
            if current_article:
                article_text += line + " "

    if current_article:
        if article_title.strip() and article_text.strip():
            structure[current_section]["articles"].append({
                'pre_title': pre_title if pre_title else "",
                'title': article_title if article_title else "",
                'article_text': article_text.strip()
            })
        else:
            if current_article:
                article_text += line + " "
                structure[current_section]["articles"].append({
                    'pre_title': pre_title if pre_title else "",
                    'title': article_title if article_title else "",
                    'article_text': article_text.strip()
                })
        
    return structure


def parse_text_into_structure_with_intro(text):
    structure = parse_text_into_structure(text)

    first_section_key = next((key for key in structure.keys() if key.startswith("I.")), None)
    if first_section_key and "articles" in structure[first_section_key]:
        first_article = structure[first_section_key]["articles"][0]  # get the first article
        if first_article:
            structure["Introduction"]["article_text"] = first_article["article_text"]
            
            structure[first_section_key]["articles"].pop(0)
            
            if not structure[first_section_key]["articles"]:
                del structure[first_section_key]

    return structure


def save_to_json(parsed_structure, filename="output.json"):
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(parsed_structure, json_file, ensure_ascii=False, indent=4)
