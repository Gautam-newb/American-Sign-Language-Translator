"""
Improved Model Training Script

This script provides an enhanced training pipeline with:
- Data normalization using StandardScaler
- Improved model architecture with batch normalization
- Advanced callbacks (early stopping, model checkpointing, learning rate reduction)
- Better data handling for inconsistent CSV formats
- Model evaluation and scaler saving for inference
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import csv

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
NUM_CLASSES = 30
INPUT_SIZE = 42  # 21 keypoints * 2 (x, y)
BATCH_SIZE = 64
EPOCHS = 200
VALIDATION_SPLIT = 0.2
INITIAL_LEARNING_RATE = 0.001

# Create model directory if it doesn't exist
os.makedirs('model/keypoint_classifier', exist_ok=True)

# Load and preprocess data
print("Loading data...")
X = []
y = []

# Load data manually to handle inconsistent columns
with open('model/keypoint_classifier/keypoint.csv', 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        # First column is the label
        label = int(row[0])
        # Next 42 columns are the features (21 keypoints * 2 coordinates)
        features = [float(x) for x in row[1:43]]  # Only take the first 42 features
        X.append(features)
        y.append(label)

X = np.array(X)
y = np.array(y)

# Print dataset statistics
print(f"\nDataset Statistics:")
print(f"Total samples: {len(X)}")
for i in range(NUM_CLASSES):
    count = np.sum(y == i)
    print(f"Class {i}: {count} samples")

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Create improved model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(INPUT_SIZE,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model

# Create and compile model
model = create_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'model/keypoint_classifier/keypoint_classifier.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
]

# Train the model
print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT,
    callbacks=callbacks,
    verbose=1
)

# Evaluate the model
print("\nEvaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# Save the scaler for preprocessing new data
import joblib
joblib.dump(scaler, 'model/keypoint_classifier/keypoint_scaler.pkl')

print("\nTraining completed! Model and scaler have been saved.") 