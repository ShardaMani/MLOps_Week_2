import pandas as pd
import sys
import os
# import yaml  <- No longer needed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

# --- 1. Define Parameters Directly ---
# Parameters are now hardcoded in the script
test_size = 0.2
random_state = 42
epochs = 50
batch_size = 16
dense_1_nodes = 16
dropout_rate = 0.2
dense_2_nodes = 8
optimizer = 'adam'

# --- 2. Load and Prepare the Dataset ---
data_path = os.path.join('data', 'iris.csv')
model_path = 'iris_model.h5'
metrics_path = "metrics.csv"

# ... (rest of the data loading code is the same)
# ...
X_train, X_val, y_train, y_val = train_test_split(
    X, y_one_hot, test_size=test_size, random_state=random_state
)
# ... (rest of the data preparation code is the same)


# --- 3. Define and Compile the Model ---
model = Sequential()
model.add(Dense(dense_1_nodes, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(dropout_rate))
model.add(Dense(dense_2_nodes, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=optimizer,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ... (rest of the script is the same, just use the variables)
# ...
model.fit(X_train, y_train,
          epochs=epochs,
          batch_size=batch_size,
          #...
          )
