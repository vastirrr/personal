import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Load the dataset
dataset = pd.read_csv('cancer.csv')

# Separate features (x) and target variable (y)
x = dataset.drop(columns=["diagnosis(1=m, 0=b)"]) #everything exept this column
y = dataset["diagnosis(1=m, 0=b)"] #just this column

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# test_size=0.2 <=> 20% of the data is going to be tested in set

# Build the neural network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, input_shape=(x_train.shape[1],), activation='sigmoid'),  # Input layer with 256 neurons
    tf.keras.layers.Dense(256, activation='sigmoid'),  # Hidden layer with 256 neurons
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron (binary classification)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#accuracy bcz we want to clasify as many tumors as possible

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=1)

#evaluate
model.evaluate(x_test, y_test)