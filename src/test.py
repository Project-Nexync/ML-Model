import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from sentence_transformers import SentenceTransformer

# --- Parameters ---
TOP_N = 3  # Number of members to suggest

# --- Step 1: Load trained model ---
model = load_model('../model_artifacts/task_member_model.h5')
print("✅ Loaded trained model")

# --- Step 2: Load member features and metadata ---
member_features = np.load('../data/features/member_features.npy')
member_meta = pd.read_csv('../data/features/member_metadata.csv')

# --- Step 3: Load task feature model (SentenceTransformer) ---
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Step 4: Define new task ---
new_task = "Build responsive login page with authentication"

# --- Step 5: Encode task description ---
task_embedding = sentence_model.encode([new_task], show_progress_bar=False)

# --- Step 6: Combine task features with skills/category (simulate for example) ---
# If your tasks also have skills/category, you should build full task_features
# For simplicity, here we use embedding only and pad/concatenate to match dimensions
# Assuming task_features.npy is 629-d
dummy_task_features = np.zeros((1, 629))
dummy_task_features[0, :len(task_embedding[0])] = task_embedding[0]

# --- Step 7: Prepare pair features for all members ---
num_members = member_features.shape[0]
task_repeated = np.repeat(dummy_task_features, num_members, axis=0)
pair_features = np.hstack([task_repeated, member_features])

# --- Step 8: Scale features using same StandardScaler as training ---
scaler = StandardScaler()
pair_features = scaler.fit_transform(pair_features)  # replace with saved scaler if available

# --- Step 9: Predict probabilities ---
probs = model.predict(pair_features, batch_size=32).flatten()

# --- Step 10: Select top-N members ---
top_indices = probs.argsort()[-TOP_N:][::-1]
top_members = member_meta.iloc[top_indices]
top_members = top_members.copy()
top_members['probability'] = probs[top_indices]

print(f"\nTop-{TOP_N} suggested members for task: '{new_task}'")
print(top_members[['member_id', 'name', 'skills', 'tasks_done', 'tasks_assigned', 'probability']])
