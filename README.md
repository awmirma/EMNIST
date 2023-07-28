# EMNIST Image Classification Project

## Overview

This is a simple project that demonstrates how to train a Convolutional Neural Network (CNN) to classify handwritten letters and digits from the Extended Modified National Institute of Standards and Technology (EMNIST) dataset. The CNN model will be trained using TensorFlow and Keras.

## Prerequisites

To run this project, you need to have the following installed on your system:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- emnist (install using `pip install emnist`)

## Project Structure

The project is organized into the following files:

- `train_model.py`: Contains the CNN model definition, data loading, preprocessing, training, and evaluation.
- `processes.py`: Contains functions for data loading and other processes.

## Getting Started

1. Clone this repository to your local machine:

```
git clone https://github.com/awmirma/EMNIST.git
cd EMNIST
```

2. Install the required packages:

```
pip install tensorflow keras numpy emnist
```

3. Run the `train_model.py` script to train and evaluate the model:

```
python train_model.py
```


The script will download the EMNIST dataset, preprocess the data, define the CNN model, train it, and evaluate its performance on the test dataset. It will also stop training early if the validation accuracy does not improve for a certain number of epochs, using early stopping.

## Results

After running the `train_model.py` script, you will see the training progress and the test accuracy printed in the console. The best model with the highest validation accuracy will be saved to 'best_emnist_cnn_model.h5'.

## Further Improvements

- You can experiment with different CNN architectures, hyperparameters, and optimization techniques to improve the model's performance.
- Try data augmentation to increase the dataset size and potentially enhance the model's generalization.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to use and modify this project for your learning and experimentation.

## Acknowledgments

- The EMNIST dataset is provided by NIST: https://www.nist.gov/itl/products-and-services/emnist-dataset
- Thanks to TensorFlow and Keras for providing the deep learning tools used in this project.
