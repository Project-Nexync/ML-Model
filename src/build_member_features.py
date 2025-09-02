import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
import os

# --- Step 1: Load member CSV ---
df = pd.read_csv('../data/processed/members_with_id.csv')

# --- Step 2: Convert skills to list ---
df['skills_list'] = df['skills'].apply(lambda x: [s.strip() for s in x.split(',')])

# --- Step 3: Multi-hot encode skills ---
# Use the SAME MultiLabelBinarizer fitted on task skills if you have it
mlb = MultiLabelBinarizer()
skills_encoded = mlb.fit_transform(df['skills_list'])
print("Skills encoded shape:", skills_encoded.shape)

# --- Step 4: Include numeric features ---
numeric_features = df[['tasks_done', 'tasks_assigned']].values
print("Numeric features shape:", numeric_features.shape)

# --- Step 5: Combine all features ---
member_features = np.hstack([skills_encoded, numeric_features])
print("Final member features shape:", member_features.shape)

# --- Step 6: Save features and metadata ---
os.makedirs('../data/features', exist_ok=True)
np.save('../data/features/member_features.npy', member_features)

metadata = df[['member_id', 'name', 'skills', 'tasks_done', 'tasks_assigned']]
metadata.to_csv('../data/features/member_metadata.csv', index=False)

print("✅ Saved member_features.npy and member_metadata.csv in data/features/")
