# app/model.py

import joblib
import sqlite3
import os

import torch
import onnxruntime as ort
from transformers import AutoTokenizer

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, '..')

# Load SVD model
svd_model = joblib.load(os.path.join(MODEL_DIR, 'svd_model.joblib'))

# Load Sentence-BERT ONNX model
sbert_session = ort.InferenceSession(os.path.join(MODEL_DIR, 'sbert_model.onnx'))

# Load Sentence-BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, 'sbert_tokenizer'))

# Load TF-IDF Vectorizer
tfidf_vectorizer = joblib.load(os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib'))

# Load SQLite cache (if applicable)
db_path = os.path.join(MODEL_DIR, 'course_embeddings.db')
conn = sqlite3.connect(db_path)
