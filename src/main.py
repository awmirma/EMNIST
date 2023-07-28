import numpy as np
from keras import layers, models

# Preprocess the EMNIST data
def preprocess_emnist_data(train_images, test_images):
    # Normalize pixel values to the range [0, 1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Flatten the images (convert 28x28 images to 1D arrays)
    train_images = train_images.reshape(train_images.shape[0], 28 * 28)
    test_images = test_images.reshape(test_images.shape[0], 28 * 28)

    return train_images, test_images

if __name__ == '__main__':
    # Load the EMNIST data
    train_images, train_labels, test_images, test_labels = load_emnist_data()

    # Preprocess the EMNIST data
    train_images, test_images = preprocess_emnist_data(train_images, test_images)

    # # Print the shapes of the preprocessed data to verify
    # print("Train images shape:", train_images.shape)
    # print("Train labels shape:", train_labels.shape)
    # print("Test images shape:", test_images.shape)
    # print("Test labels shape:", test_labels.shape)


def create_emnist_cnn():
    model = models.Sequential([
        layers.Reshape((28, 28, 1), input_shape=(28 * 28,)),  # Reshape 1D input to 2D
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(62, activation='softmax')  # 62 classes (26 uppercase letters + 26 lowercase letters + 10 digits)
    ])
    return model

model = create_emnist_cnn()
# model.summary()


# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(train_images, train_labels, epochs=3, batch_size=128, validation_split=0.1)

# Optional
# Prediction
# predictions = model.predict(test_images[:10])
# Save the model
# model.save('emnist_cnn_model.h5')
