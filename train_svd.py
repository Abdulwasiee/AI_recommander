from surprise import SVD, Dataset, Reader
import pandas as pd
import joblib

# Sample user-course ratings
data = pd.DataFrame({
    'user_id': ['U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U4', 'U5', 'U5'],
    'course_id': ['C001', 'C002', 'C001', 'C003', 'C003', 'C004', 'C004', 'C005', 'C002', 'C005'],
    'rating': [5, 4, 4, 3, 5, 2, 3, 5, 4, 5]
})

reader = Reader(rating_scale=(1, 5))
dataset = Dataset.load_from_df(data[['user_id', 'course_id', 'rating']], reader)
trainset = dataset.build_full_trainset()

# Train SVD model
svd = SVD()
svd.fit(trainset)

# Save to file
joblib.dump(svd, "svd_model.joblib")
print("✅ SVD model saved as 'svd_model.joblib'")
