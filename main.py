import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
data = fetch_openml('mnist_784')
X = data.data
y = data.target.astype(int)  # Convert target to integers

# Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the neural network model
def create_model():
    model = Sequential([
        Dense(16, input_dim=784),  # Input layer with 784 neurons (784 features per image)
        Activation('relu'),        # Relu activation function
        Dense(8),                  # Hidden layer with 8 neurons
        Activation('relu'),        # Relu activation function
        Dense(10),                 # Output layer with 10 neurons (10 classes)
        Activation('softmax')      # Softmax activation function
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create the model
model = create_model()

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Predict using the model
predictions = model.predict(X_test)

# Example prediction on a test image
image = X_test[:1]  # Assuming the first test image
prediction = model.predict(image)

print(f"Predicted number: {np.argmax(prediction)}")
