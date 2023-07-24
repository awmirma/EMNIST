import numpy as np
from emnist import extract_training_samples, extract_test_samples

# Download and load the EMNIST dataset
def load_emnist_data():
    # Extract training samples (images and labels)
    train_images, train_labels = extract_training_samples('byclass')
    
    # Extract test samples (images and labels)
    test_images, test_labels = extract_test_samples('byclass')

    return train_images, train_labels, test_images, test_labels

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

    # Print the shapes of the preprocessed data to verify
    print("Train images shape:", train_images.shape)
    print("Train labels shape:", train_labels.shape)
    print("Test images shape:", test_images.shape)
    print("Test labels shape:", test_labels.shape)
