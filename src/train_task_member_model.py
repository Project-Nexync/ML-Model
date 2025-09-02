import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# --- Step 1: Load simulated dataset ---
X = np.load('../data/features/task_member_pairs.npy')
y = np.load('../data/features/task_member_labels.npy')

print("Loaded features:", X.shape)
print("Loaded labels:", y.shape)

# --- Step 2: Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Step 3: Scale features ---
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --- Step 4: Build MLP model ---
model = Sequential([
    Dense(256, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # probability output
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- Step 5: Train ---
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1
)

# --- Step 6: Evaluate ---
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.4f}")

# --- Step 7: Save model ---
os.makedirs('../model_artifacts', exist_ok=True)
model.save('../model_artifacts/task_member_model.h5')
print("✅ Model saved to model_artifacts/task_member_model.h5")
