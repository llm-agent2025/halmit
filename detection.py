import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# load M3E-base
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_data_from_json(file_path):

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_question_embeddings(questions):

    return model.encode(questions)

def calculate_centroid(vectors):

    return np.mean(vectors, axis=0)

def detect_hallucination(query, data, threshold=0.8):

    questions = [entry['question'] for entry in data]
    question_embeddings = get_question_embeddings(questions)
    
  
    query_embedding = model.encode([query])
    
 
    similarities = cosine_similarity(query_embedding, question_embeddings).flatten()

    top_indices = similarities.argsort()[-3:][::-1]  # 获取前三个最相似的问题索引
    
    top_vectors = question_embeddings[top_indices]
    top_similarities = similarities[top_indices]
    

    centroid = calculate_centroid(top_vectors)
    

    centroid_similarity = cosine_similarity(query_embedding, [centroid]).flatten()[0]
    
    print(f"Top 3 similar questions' similarities: {top_similarities}")
    print(f"Centroid similarity with query: {centroid_similarity}")
    

    if centroid_similarity > threshold:
        print("Hallucination detected!")
    else:
        print("No hallucination detected.")

if __name__ == "__main__":

    file_path = 'data.json'  
    data = load_data_from_json(file_path)

    query = "  "
    
    detect_hallucination(query, data)
