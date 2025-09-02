import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sentence_transformers import SentenceTransformer

# --- Step 1: Load processed tasks ---
df = pd.read_csv('../data/processed/tasks_with_id.csv')

# --- Step 2: Encode description into embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')
task_descriptions = df['description'].tolist()
task_embeddings = model.encode(task_descriptions, show_progress_bar=True)

print("Embeddings shape:", task_embeddings.shape)  # (20122, 384)

# --- Step 3: Encode skills (multi-hot) ---
df['skills_list'] = df['skills'].apply(lambda x: [s.strip() for s in x.split(',')])
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['skills_list'])
print("Skills shape:", skills_encoded.shape)  # (20122, 232)

# --- Step 4: Encode category (one-hot) ---
category_encoded = pd.get_dummies(df['category'])
print("Category shape:", category_encoded.shape)  # (20122, 13)

# --- Step 5: Combine into one feature matrix ---
task_features = np.hstack([
    task_embeddings,        # (20122, 384)
    skills_encoded,         # (20122, 232)
    category_encoded.values # (20122, 13)
])
print("Final task features shape:", task_features.shape)  # (20122, 629)

# --- Step 6: Save results ---
np.save('../data/features/task_features.npy', task_features)

metadata = df[['task_id', 'description', 'skills', 'category']]
metadata.to_csv('../data/features/task_metadata.csv', index=False)

print("✅ Saved task_features.npy and task_metadata.csv in data/features/")
