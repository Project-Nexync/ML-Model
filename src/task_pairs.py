import numpy as np
import pandas as pd

# Load task and member features
task_features = np.load('../data/features/task_features.npy')
member_features = np.load('../data/features/member_features.npy')

task_meta = pd.read_csv('../data/features/task_metadata.csv')
member_meta = pd.read_csv('../data/features/member_metadata.csv')

np.random.seed(42)

pairs = []
labels = []

for i, task_vec in enumerate(task_features):
    # Option 1: random member assignment
    assigned_member_idx = np.random.randint(len(member_features))
    
    for j, member_vec in enumerate(member_features):
        # Combine task + member features
        combined_vec = np.hstack([task_vec, member_vec])
        pairs.append(combined_vec)
        
        # Label: 1 if assigned, 0 otherwise
        label = 1 if j == assigned_member_idx else 0
        labels.append(label)

pairs = np.array(pairs)
labels = np.array(labels)

print("Pair features shape:", pairs.shape)
print("Labels shape:", labels.shape)

# Save for training
np.save('../data/features/task_member_pairs.npy', pairs)
np.save('../data/features/task_member_labels.npy', labels)

print("✅ Simulated task-member dataset ready")
