import numpy as np
from emnist import extract_training_samples, extract_test_samples
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

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

def preprocess_emnist_data(train_images, test_images):
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    train_images = train_images.reshape(train_images.shape[0], 28 * 28)
    test_images = test_images.reshape(test_images.shape[0], 28 * 28)

    return train_images, test_images

def train_emnist_model(train_images, train_labels, test_images, test_labels):
    model = create_emnist_cnn()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=100, batch_size=128, validation_split=0.1)

    # Define early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_emnist_cnn_model.h5', monitor='val_accuracy', save_best_only=True)

    history = model.fit(train_images, train_labels, epochs=100, batch_size=128, validation_split=0.1,
                        callbacks=[early_stopping, model_checkpoint])

    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_accuracy)
    print("Test loss:", test_loss)

    return model, history

if __name__ == '__main__':
    train_images, train_labels = extract_training_samples('byclass')
    test_images, test_labels = extract_test_samples('byclass')
    train_images, test_images = preprocess_emnist_data(train_images, test_images)

    trained_model, training_history = train_emnist_model(train_images, train_labels, test_images, test_labels)
