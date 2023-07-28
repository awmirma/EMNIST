# processes.py

from emnist import extract_training_samples, extract_test_samples

def load_emnist_data():
    train_images, train_labels = extract_training_samples('byclass')
    test_images, test_labels = extract_test_samples('byclass')
    return train_images, train_labels, test_images, test_labels
