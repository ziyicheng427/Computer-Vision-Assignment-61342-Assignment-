import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Function to load and preprocess data
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Fashion-MNIST images are grayscale; need to expand dimensions to use with Conv2D
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    # Convert labels to categorical
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Adjust AlexNet model for the Fashion-MNIST dataset
def alexnet(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Conv2D(48, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(192, (3, 3), padding='same', activation='relu'),
        Conv2D(192, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Load and preprocess Fashion-MNIST data
x_train_fashion, y_train_fashion, x_test_fashion, y_test_fashion = load_and_preprocess_data()

# Train on Fashion-MNIST
model_fashion = alexnet()
model_fashion.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history_fashion = model_fashion.fit(x_train_fashion, y_train_fashion, batch_size=64, epochs=30, validation_split=0.1, verbose=1)

# Plotting the training results
plt.figure(figsize=(8, 6))
plt.plot(history_fashion.history['accuracy'], label='Training Accuracy')
plt.plot(history_fashion.history['val_accuracy'], label='Validation Accuracy')
plt.title('Fashion-MNIST Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
