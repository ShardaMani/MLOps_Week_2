'''
This script trains a simple neural network on the Iris dataset.
It is designed to be used within a DVC pipeline.

Expected Directory Structure:
/
|-- data/
|   |-- iris.csv
|-- train_iris.py
'''
import pandas as pd
import sys
import os
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# --- 1. Load Parameters ---
# Load parameters from a file (good practice for DVC)
with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

# --- 2. Load and Prepare the Dataset ---
# DVC will track this data file
data_path = os.path.join('data', 'iris.csv')
model_path = 'iris_model.h5'
metrics_path = "metrics.csv"

# Column names for the Iris dataset
col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_df = pd.read_csv(data_path, header=None, names=col_names)

print("Dataset loaded successfully.")
print("First 5 rows of the dataset:")
print(iris_df.head())

# Separate features (X) and labels (y)
X = iris_df.drop('species', axis=1)
y = iris_df['species']

# Convert categorical labels to numerical and then to one-hot encoding
# e.g., 'Iris-setosa' -> 0, 'Iris-versicolor' -> 1, 'Iris-virginica' -> 2
y_encoded = pd.Categorical(y).codes
y_one_hot = to_categorical(y_encoded, num_classes=3)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y_one_hot, test_size=params['train']['test_size'], random_state=params['train']['random_state']
)

# Scale the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")


# --- 3. Define and Compile the Model ---
model = Sequential()
# Input layer must match the number of features (4)
model.add(Dense(params['model']['dense_1'], activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(params['model']['dropout']))
model.add(Dense(params['model']['dense_2'], activation='relu'))
# Output layer has 3 neurons (for 3 classes) and softmax activation
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=params['model']['optimizer'],
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- 4. Train the Model ---
print("\nStarting model training...")
model.fit(X_train, y_train,
          epochs=params['train']['epochs'],
          batch_size=params['train']['batch_size'],
          validation_data=(X_val, y_val),
          verbose=0,  # TQDM will handle the progress bar
          callbacks=[TqdmCallback(verbose=1), CSVLogger(metrics_path)])

print("Training finished.")


# --- 5. Save the Model ---
# The final model file will be tracked by DVC
model.save(model_path)
print(f"Model saved to {model_path}")
