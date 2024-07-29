import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np

# Load your dataset
data = pd.read_csv("F:\\summer\\Recommendation\\advertising.csv")

# Features and target for content-based filtering
X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = data['Clicked on Ad']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
numeric_features = ['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)])

# Model pipeline for content-based filtering
content_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Fit the content model
content_model.fit(X_train, y_train)

# Save the content-based model
joblib.dump(content_model, 'content_model.pkl')

# For collaborative filtering, we'll use a basic user-item interaction matrix
interaction_data = data[['Male', 'Ad Topic Line', 'Clicked on Ad']]

# Create a user-item matrix
interaction_data.loc[:, 'Clicked on Ad'] = interaction_data['Clicked on Ad'].astype(float)
user_item_matrix = interaction_data.pivot_table(index='Male', columns='Ad Topic Line', values='Clicked on Ad', aggfunc='mean').fillna(0)

# Example collaborative filtering using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
user_similarity = cosine_similarity(user_item_matrix)

# Save the user similarity matrix
np.save('user_similarity.npy', user_similarity)
