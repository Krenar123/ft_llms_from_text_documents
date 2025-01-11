import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the question-answer pairs from the JSON file
with open('qa_pairs.json', 'r') as f:
    data = json.load(f)

# Function to check if the question makes sense
def check_similarity(question, answer, threshold=0.7):
    # Combine the question and answer to calculate similarity
    texts = [question, answer]
    
    # Vectorize the texts using TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(texts)
    cosine_sim = cosine_similarity(vectorizer[0:1], vectorizer[1:2])
    
    # Check if the similarity is above the threshold
    return cosine_sim[0][0] >= threshold

# Check each pair in the loaded data
for pair in data:
    question = pair['question']
    answer = pair['answer']
    
    if check_similarity(question, answer):
        print(f"Question: '{question}' makes sense with the answer.")
    else:
        print(f"Question: '{question}' does NOT make sense with the answer.")
