import numpy as np

IMAGE_SIZE = 28


def load_train_data(data_path, validation_size = 500):
	 """
	 Data Format is 28x28 images
	 Load the mnist data from the csv file and return a 3D Tensor input
	 of train and validn set with 2D tensor of one hot encoding
	 """

	 train_data = np.genfromtxt(data_path, delimiter = ',', dtype=np.float)
	 x_train = train_data[:, 1:]

	 y_train = train_data[:, 0]
	 y_train = (np.arange(10) == y_train[:, None]).astype(np.float32)

	 x_train, x_val, y_train, y_val = x_train[0:(len(x_train) - validation_size), :], \
	 x_train[len(x_train) - validation_size:len(x_train), :], \
	 y_train[0:(len(y_train) - validation_size), :], \
	 y_train[(len(y_train) - validation_size):len(y_train), :]
	 
	 x_train = x_train.reshape(len(x_train), IMAGE_SIZE, IMAGE_SIZE, 1)
	 x_val = x_val.reshape(len(x_val), IMAGE_SIZE, IMAGE_SIZE, 1)

	 return x_train, x_val, y_train, y_val

def load_test_data(data_path):
    """
    Load mnist test data
    :return: 3D Tensor input of train and validation set with
     2D Tensor of one hot encoded image labels
    """
    test_data = np.genfromtxt(data_path, delimiter=',', dtype=np.float32)
    x_test = test_data[:, 1:]

    y_test = np.array(test_data[:, 0])
    y_test = (np.arange(10) == y_test[:, None]).astype(np.float32)

    x_test = x_test.reshape(len(x_test), IMAGE_SIZE, IMAGE_SIZE, 1)

    return x_test, y_test