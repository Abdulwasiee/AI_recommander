# app/routes.py

from flask import Blueprint, request, jsonify
from app.recommender import hybrid_recommend
import pandas as pd
import os

api = Blueprint('api', __name__)

# Load course metadata (assumes it's saved as CSV in the root)
course_data_path = os.path.join(os.path.dirname(__file__), '..', 'coursera_course_dataset_v3.csv')
course_df = pd.read_csv(course_data_path)

@api.route("/recommend", methods=["POST"])
def get_recommendations():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        query = data.get("query")

        if not user_id or not query:
            return jsonify({"error": "user_id and query are required"}), 400

        results = hybrid_recommend(user_id, query, course_df)
        return jsonify({"recommendations": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
