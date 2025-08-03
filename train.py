# app/model/train_model.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load data
gene_df = pd.read_csv(r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\data\rNA_data.csv")
label_df = pd.read_csv(r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\data\RNA_Label.csv")

# Prepare X and y
X = gene_df.drop(columns=["Unnamed: 0"])
y = label_df["Class"]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Save label encoder for future use
joblib.dump(le, r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\model\label_encoder.pkl")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\model\scaler.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
)

# Build the model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Save model
model.save(r"C:\Users\saeem\Desktop\RNA_SEQ_for _Cancer\model\model.h5")
