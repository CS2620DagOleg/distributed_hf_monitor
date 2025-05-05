import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load heart failure dataset and showing the dataset
print("Loading heart failure dataset...")
data = pd.read_csv('hf.csv')
print(f"Dataset shape: {data.shape}")
print(data.head())

# Extracting features to be used for assessing risk, we are only using five features
# [age, serum_sodium, serum_creatinine, ejection_fraction, time (day)]
X = data[['age', 'serum_sodium', 'serum_creatinine', 'ejection_fraction', 'time']].values
y = data['DEATH_EVENT'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'hf_scaler.gz')
print("Scaler saved to hf_scaler.gz")

# Create a NN and compile
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_scaled, y_test),
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test accuracy: {accuracy:.4f}")

# Save the model
model.save('heart_failure_model.h5')
print("Model saved to heart_failure_model.h5")

# Quick test of the model
print("\nTesting model with some sample data...")

def predict_risk(age, sodium, creatinine, ef, day):
    sample = np.array([[age, sodium, creatinine, ef, day]])
    sample_scaled = scaler.transform(sample)
    probability = model.predict(sample_scaled)[0][0]
    
    if probability < 0.3:
        risk = "GREEN (Low Risk)"
    elif probability < 0.6:
        risk = "AMBER (Medium Risk)"
    else:
        risk = "RED (High Risk)"
    
    return probability, risk

# Test with different risk levels
test_cases = [
    # Low risk case (elderly but good vitals)
    (75, 140, 0.9, 60, 30),
    # Medium risk case
    (70, 134, 1.5, 38, 60),
    # High risk case
    (80, 125, 2.2, 25, 90)
]

for test_case in test_cases:
    age, sodium, creatinine, ef, day = test_case
    prob, risk = predict_risk(age, sodium, creatinine, ef, day)
    print(f"Age={age}, Na={sodium}, Creat={creatinine}, EF={ef}%, Day={day}")
    print(f"Prediction: {prob:.4f} - {risk}\n")

print("Setup complete. The monitoring System can now be run.")
