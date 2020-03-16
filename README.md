# Digit Recognizer
## A Basic Handwritten Digit Recognizer using Convolutional Neural Networks

This little project uses Tensorflow backend and is trained on the MNIST Handwritten Digits Dataset. The Test_Accuracy file has the testing accuracy of this network saved.

If you don't have the required Anaconda environment ready, you can just install Tensorflow via pip (assuming you have Python 3.5+)

```$ pip install tensorflow ```

You can download the training and testing dataset as well:

```
$ mkdir data
$ curl https://pjreddie.com/media/files/mnist_train.csv -o data/mnist_train.csv
$ curl https://pjreddie.com/media/files/mnist_test.csv -o data/mnist_test.csv
```

Then just use:

```
$ python3 train.py
```
to train the network, which will print cost function after every epoch. After 10000, evaluate the network using:
```
$ python3 evaluate.py >> Test_Accuracy.txt
```

Additionally, you can visualize the results (live loss, computational graph, etc) using Tensorboard as such:
```
$ tensorboard --logdir=graphs/ --port=6006
```
And navigate in browser to the link which is printed.
