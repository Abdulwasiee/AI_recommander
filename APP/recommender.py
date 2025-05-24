# app/recommender.py

import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from app.model import svd_model, tokenizer, sbert_session, tfidf_vectorizer, conn

def run_sbert_inference(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    input_ids = inputs['input_ids'].numpy()
    attention_mask = inputs['attention_mask'].numpy()
    
    output = sbert_session.run(
        None,
        {'input_ids': input_ids, 'attention_mask': attention_mask}
    )[0]

    embedding = np.mean(output, axis=1)  # mean pooling
    return embedding

def hybrid_recommend(user_id, query, course_df, top_n=5, collab_weight=0.6):
    # Step 1: Collaborative Filtering Scores
    collab_scores = {}
    for course_id in course_df['course_id']:
        pred = svd_model.predict(user_id, course_id)
        collab_scores[course_id] = pred.est

    # Step 2: Content-Based Scores (SBERT)
    sbert_embedding = run_sbert_inference(query)
    
    course_embeddings = []
    for _, row in course_df.iterrows():
        desc = f"{row['course_description']} {row['institution']} {row['Skills']}"
        emb = run_sbert_inference(desc)
        course_embeddings.append(emb)

    similarities = cosine_similarity(sbert_embedding, np.vstack(course_embeddings)).flatten()

    # Step 3: Combine scores
    final_scores = {}
    for idx, course_id in enumerate(course_df['course_id']):
        hybrid_score = collab_weight * collab_scores.get(course_id, 3.0) + (1 - collab_weight) * similarities[idx]
        final_scores[course_id] = hybrid_score

    # Step 4: Get Top N
    top_courses = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    result = []

    for course_id, score in top_courses:
        course_data = course_df[course_df['course_id'] == course_id].iloc[0]
        result.append({
            'course_id': course_id,
            'title': course_data['Title'],
            'institution': course_data['institution'],
            'score': round(score, 2)
        })

    return result
