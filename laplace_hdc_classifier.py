# import the necessary packages
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

# Use the GPU if available, else use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_hdc_classifier(model, testloader):
    """Test the accuracy of a classifier.

    Keyword arguments:
    model:      the classificatio model to test
    testloader: the dataloder containing the testing data
    """

    # turn the model into evaluation mode and select the criterion
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # counter for number of correct predicitons
    correct = 0

    # gradient doesn't matter in this case
    with torch.no_grad():
        # loop over the data in the testloader
        for data in testloader:
            # extract the data and labels, send to device
            inputs, labels = data[0].to(device), data[1].to(device)

            # perform the inference
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs.data, dim=1).indices

            # tally the number of correct predictions in the batch
            correct += (predicted == labels).sum().item()

    # return the accuracy of the model
    return correct / len(testloader.dataset)


def train_hdc_classifier(trainloader, num_classes, mode="binary_sgd", epochs=1):
    """Train the classifier with given training data.

    Keyword arguments:
    trainloader: data loader for training data.
    num_classes: the number of classes to categorize data into
    mode:        the mode to use for classification (default: 'binary_sgd')
    epochs:      number of times to pass over data (default: 1)
    """

    # extract the hyperdimensionality of the data
    hyperdim = np.prod(next(iter(trainloader))[0].shape[1:])

    # set model based on selected classification mode
    if mode == "binary_sgd":
        model = binary_sgd(trainloader, num_classes, epochs, hyperdim)

    elif mode == "float_sgd":
        model = float_sgd(trainloader, num_classes, epochs, hyperdim)

    elif mode == "binary_majority":
        model = binary_majority(trainloader, num_classes, hyperdim)

    elif mode == "float_majority":
        model = float_majority(trainloader, num_classes, hyperdim)

    elif mode == "cnn_sgd":
        # makes verify the shape of the data is square
        sz = next(iter(trainloader))[0].shape[1:]
        assert len(sz) == 2 and sz[0] == sz[1]

        # set the cnn_sgd classification
        model = cnn_sgd(trainloader, num_classes, epochs, sz)

    # if the string is not recognized, throw an error and crash
    else:
        print("train_hdc_classifier: invalid mode")
        sys.exit(1)

    # return the classification model
    return model


def float_sgd(trainloader, num_classes, epochs, hyperdim):
    """Stochastic gradient descent in float mode.

    Keyword arguments:
    trainloader: loader with training data
    num_classes: number of classes the data represents
    epochs:      number of times to go over the data
    hyperdim:    dimensionality of the hyperspace
    """

    # private NN to serve as classifier, which extends the NN module in torch
    class Model(torch.nn.Module):
        def __init__(
            self,
            hyperdim,
            num_classes,
        ):
            """Class initializer

            Class attributes:
            hyperdim: hyperdimensionality of hyperspace
            linear: the linear neural network for classification

            Keyword arguments:
            hyperdim: dimensionality of hyperspace
            num_classes: the number of categories into which to categorize the data
            """

            # run the initialization from parent class
            super(Model, self).__init__()

            # built the linear model w/ given parameters
            self.linear = nn.Linear(
                in_features=hyperdim, out_features=num_classes, bias=False
            )

            # set the hyperdimensionality of the model
            self.hyperdim = hyperdim

        def forward(self, x):
            """The forward pass for the model"""

            # perform the the pass over the data
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = x / self.hyperdim ** 0.5  # normalize variance of output

            # return the data
            return x

    # Create the model defined above and put it on the GPU
    model = Model(hyperdim, num_classes).to(device)

    # set the classification criteria
    criterion = torch.nn.CrossEntropyLoss()

    # set the learning rate
    alpha = 0.01

    # choose the optimizer to be the Adam model
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # set the model to be in training mode
    model.train()

    # for the number of desired passes over the data
    for epoch in tqdm(range(epochs)):

        # loop over the data in the train loader
        for inputs, labels in trainloader:

            # Put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # return the model
    return model


def binary_sgd(trainloader, num_classes, epochs, hyperdim):
    """Stochastic gradient descent in binary mode

    This function is largely the same as the float mode.

    Keyword arguments:
    trainloader: loader with training data
    num_classes: number of classes the data represents
    epochs:      number of times to go over the data
    hyperdim:    dimensionality of the hyperspace
    """

    # private NN to serve as classifier, which extends the NN module in torch
    class Model(torch.nn.Module):
        """Class initializer

        Class attributes:
        hyperdim: hyperdimensionality of hyperspace
        linear: the linear neural network for classification

        Keyword arguments:
        hyperdim: dimensionality of hyperspace
        num_classes: the number of categories into which to categorize the data
        """

        def __init__(
            self,
            hyperdim,
            num_classes,
        ):
            """Class initializer

            Class attributes:
            hyperdim: hyperdimensionality of hyperspace
            linear: the linear neural network for classification

            Keyword arguments:
            hyperdim: dimensionality of hyperspace
            num_classes: the number of categories into which to categorize the data
            """
            # run the initialization from parent class
            super(Model, self).__init__()

            # built the linear model w/ given parameters
            self.linear = nn.Linear(
                in_features=hyperdim, out_features=num_classes, bias=False
            )

            # set the hyperdimensionality of the model
            self.hyperdim = hyperdim

        def forward(self, x):
            """The forward pass for the model"""

            # perform the forward pass transforms
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = x / self.hyperdim ** 0.5  # normalize variance of output

            # return the data
            return x

    # set criterion and learning rate
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.01  # learning rate

    # make a model object on the GPU
    model = Model(hyperdim, num_classes).to(device)

    # choose the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # set model to train mode
    model.train()

    # for the number of desired passes over the data
    for epoch in tqdm(range(epochs)):

        # loop over the data in the train loader
        for inputs, labels in trainloader:

            # Put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # set any weights above 1.0 to 1.0, and below -1.0 to -1.0
            with torch.no_grad():
                model.linear.weight.data = torch.clamp(
                    model.linear.weight.data, min=-1, max=1
                )

    # after the step, training passes, make all the weights in {+-1}
    with torch.no_grad():
        model.linear.weight.data = torch.sign(model.linear.weight.data)

    # return the model
    return model


def binary_majority(trainloader, num_classes, hyperdim):
    """Classifier which uses majority vote in each dimension of hyperspace"""

    # set up empty tensors for accuracy calculation
    A = torch.zeros((num_classes, hyperdim), dtype=torch.float)
    counts = torch.zeros((num_classes, 1), dtype=torch.float)

    # loop over the training data
    for i, data in enumerate(trainloader, 0):
        # loop over the number of classes
        for j in range(num_classes):

            # idx indicates which data in the batch are of class j
            idx = data[1] == j

            # copy the image data in the batch
            inputs = data[0].clone()

            # reshape the data
            inputs = inputs.view(inputs.shape[0], -1)

            # update the matrix A with the votes from input
            A[j, :] = A[j, :] + torch.sum(1 - 2 * inputs[idx, :], axis=0)

            # update the counts tensor with the votes
            counts[j] = counts[j] + torch.sum(idx)

    # transform the data in A to determine the majority vote in each hyperdimension
    A = A / counts
    A = torch.sign(A)
    A = A.to(device)

    # class to act as classifier machine
    class Model(torch.nn.Module):
        def __init__(self, A):
            super(Model, self).__init__()
            self.A = A

        def forward(self, x):
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = torch.matmul(x, self.A.T)
            return x

    # construct the classification model
    model = Model(A).to(device)

    # return the classification model
    return model


def float_majority(trainloader, num_classes, hyperdim):
    """Classifier using majority vote in float mode.

    This is largely the same as the binary_majority mode above
    """

    # set up empty tensors for accuracy calculation
    A = torch.zeros((num_classes, hyperdim), dtype=torch.float)
    counts = torch.zeros((num_classes, 1), dtype=torch.float)

    # loop over the training data
    for i, data in enumerate(trainloader, 0):
        # loop over the number of classes
        for j in range(num_classes):
            # idx indicates which data in the batch are of class j
            idx = data[1] == j

            # copy the image data in the batch
            inputs = data[0].clone()

            # reshape the data
            inputs = inputs.view(inputs.shape[0], -1)

            # update the matrix A with the votes from input
            A[j, :] = A[j, :] + torch.sum(1 - 2 * inputs[idx, :], axis=0)

            # update the counts tensor with the votes
            counts[j] = counts[j] + torch.sum(idx)

    # transform the data in A to determine the majority vote in each hyperdimension\
    # the only difference between binary mode is that we don't take the sign,
    # so this ends up just being the mean vote in each hyperdimension.
    A = A / counts
    A = A.to(device)

    # class to act as classifier machine
    class Model(torch.nn.Module):
        def __init__(self, A):
            super(Model, self).__init__()
            self.A = A

        def forward(self, x):
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = torch.matmul(x, self.A.T)
            return x

    # construct the classification model
    model = Model(A).to(device)

    # return the classification model
    return model


def cnn_sgd(trainloader, num_classes, epochs, sz):
    """Classifier using convolutional neural network.

    Keyword arguments:
    trainloader: the training data
    num_classes: number of classes to classify
    epochs:      number of times to run over the data
    sz:          size of one side of the (square) input data
    """

    # make two copies of sz
    n1, n2 = sz

    class BasicCNN(nn.Module):
        """Class for the CNN classifier"""

        def __init__(self, n1, n2):
            """Initializer for the CNN classifier

            Class attributes:
            N: hyperdimensionality of the output data
            conv: convolutional neural network
            pool: max pooling transformation
            linear: linear classifier
            """

            # run the initializer for the parent class
            super(BasicCNN, self).__init__()

            # fix some model parameters
            kernel_size = 3
            stride = 1
            padding = 0
            out_channels = 16
            pool_kernel_size = 2
            pool_stride = 2

            # calculate the necessary dimensions of the output data
            m1 = (n1 - kernel_size + 2 * padding) // stride + 1
            m2 = (n2 - kernel_size + 2 * padding) // stride + 1
            k1 = (m1 - pool_kernel_size) // pool_stride + 1
            k2 = (m2 - pool_kernel_size) // pool_stride + 1
            N = k1 * k2 * out_channels

            # set they hyperdimensionality
            self.N = N

            # make the convolutional layer
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            # make the pooling layer
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

            # make the linear layer
            self.linear = nn.Linear(in_features=N, out_features=10)

            # set these dimensions
            self.n1 = n1
            self.n2 = n2

        def forward(self, x):
            """forward pass for the CNN classifier"""

            # reshape the data and transform to be +-1
            x = x.view(-1, 1, self.n1, self.n2)
            x = (1 - 2 * x).type(torch.float)

            # perform the convolution and pooling
            x = self.conv(x)
            x = self.pool(x)

            # reshape again
            x = x.view(x.size(0), -1)

            # apply the linear layer and normalize
            x = self.linear(x) / self.N ** 0.5

            # return teh data
            return x

    # set the classification criteria
    criterion = torch.nn.CrossEntropyLoss()

    # set the learning rate and build the classifier with the model defined above
    alpha = 0.01
    model = BasicCNN(n1, n2).to(device)

    # select an optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # debug
    # print(epochs)

    # set model to training mode
    model.train()

    # for the number of epochs selected for the training
    for epoch in tqdm(range(epochs)):

        # loop over the data in the trainloader
        for i, data in enumerate(trainloader, 0):

            # copy the data and labels to the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # reset the gradients
            optimizer.zero_grad()

            # pass over the model
            outputs = model(inputs)

            # find the loss according to the selected criterion
            loss = criterion(outputs, labels)

            # perform the backwards and forwards step of the training
            loss.backward()
            optimizer.step()

    # return the model
    return model
# import the necessary packages
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import time
from tqdm import tqdm

# Use the GPU if available, else use the CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_hdc_classifier(model, testloader):
    """Test the accuracy of a classifier.

    Keyword arguments:
    model:      the classificatio model to test
    testloader: the dataloder containing the testing data
    """

    # turn the model into evaluation mode and select the criterion
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    # counter for number of correct predicitons
    correct = 0

    # gradient doesn't matter in this case
    with torch.no_grad():
        # loop over the data in the testloader
        for data in testloader:
            # extract the data and labels, send to device
            inputs, labels = data[0].to(device), data[1].to(device)

            # perform the inference
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predicted = torch.max(outputs.data, dim=1).indices

            # tally the number of correct predictions in the batch
            correct += (predicted == labels).sum().item()

    # return the accuracy of the model
    return correct / len(testloader.dataset)


def train_hdc_classifier(trainloader, num_classes, mode="binary_sgd", epochs=1):
    """Train the classifier with given training data.

    Keyword arguments:
    trainloader: data loader for training data.
    num_classes: the number of classes to categorize data into
    mode:        the mode to use for classification (default: 'binary_sgd')
    epochs:      number of times to pass over data (default: 1)
    """

    # extract the hyperdimensionality of the data
    hyperdim = np.prod(next(iter(trainloader))[0].shape[1:])

    # set model based on selected classification mode
    if mode == "binary_sgd":
        model = binary_sgd(trainloader, num_classes, epochs, hyperdim)

    elif mode == "float_sgd":
        model = float_sgd(trainloader, num_classes, epochs, hyperdim)

    elif mode == "binary_majority":
        model = binary_majority(trainloader, num_classes, hyperdim)

    elif mode == "float_majority":
        model = float_majority(trainloader, num_classes, hyperdim)

    elif mode == "cnn_sgd":
        # makes verify the shape of the data is square
        sz = next(iter(trainloader))[0].shape[1:]
        assert len(sz) == 2 and sz[0] == sz[1]

        # set the cnn_sgd classification
        model = cnn_sgd(trainloader, num_classes, epochs, sz)

    # if the string is not recognized, throw an error and crash
    else:
        print("train_hdc_classifier: invalid mode")
        sys.exit(1)

    # return the classification model
    return model


def float_sgd(trainloader, num_classes, epochs, hyperdim):
    """Stochastic gradient descent in float mode.

    Keyword arguments:
    trainloader: loader with training data
    num_classes: number of classes the data represents
    epochs:      number of times to go over the data
    hyperdim:    dimensionality of the hyperspace
    """

    # private NN to serve as classifier, which extends the NN module in torch
    class Model(torch.nn.Module):
        def __init__(
            self,
            hyperdim,
            num_classes,
        ):
            """Class initializer

            Class attributes:
            hyperdim: hyperdimensionality of hyperspace
            linear: the linear neural network for classification

            Keyword arguments:
            hyperdim: dimensionality of hyperspace
            num_classes: the number of categories into which to categorize the data
            """

            # run the initialization from parent class
            super(Model, self).__init__()

            # built the linear model w/ given parameters
            self.linear = nn.Linear(
                in_features=hyperdim, out_features=num_classes, bias=False
            )

            # set the hyperdimensionality of the model
            self.hyperdim = hyperdim

        def forward(self, x):
            """The forward pass for the model"""

            # perform the the pass over the data
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = x / self.hyperdim ** 0.5  # normalize variance of output

            # return the data
            return x

    # Create the model defined above and put it on the GPU
    model = Model(hyperdim, num_classes).to(device)

    # set the classification criteria
    criterion = torch.nn.CrossEntropyLoss()

    # set the learning rate
    alpha = 0.01

    # choose the optimizer to be the Adam model
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # set the model to be in training mode
    model.train()

    # for the number of desired passes over the data
    for epoch in tqdm(range(epochs)):

        # loop over the data in the train loader
        for inputs, labels in trainloader:

            # Put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # return the model
    return model


def binary_sgd(trainloader, num_classes, epochs, hyperdim):
    """Stochastic gradient descent in binary mode

    This function is largely the same as the float mode.

    Keyword arguments:
    trainloader: loader with training data
    num_classes: number of classes the data represents
    epochs:      number of times to go over the data
    hyperdim:    dimensionality of the hyperspace
    """

    # private NN to serve as classifier, which extends the NN module in torch
    class Model(torch.nn.Module):
        """Class initializer

        Class attributes:
        hyperdim: hyperdimensionality of hyperspace
        linear: the linear neural network for classification

        Keyword arguments:
        hyperdim: dimensionality of hyperspace
        num_classes: the number of categories into which to categorize the data
        """

        def __init__(
            self,
            hyperdim,
            num_classes,
        ):
            """Class initializer

            Class attributes:
            hyperdim: hyperdimensionality of hyperspace
            linear: the linear neural network for classification

            Keyword arguments:
            hyperdim: dimensionality of hyperspace
            num_classes: the number of categories into which to categorize the data
            """
            # run the initialization from parent class
            super(Model, self).__init__()

            # built the linear model w/ given parameters
            self.linear = nn.Linear(
                in_features=hyperdim, out_features=num_classes, bias=False
            )

            # set the hyperdimensionality of the model
            self.hyperdim = hyperdim

        def forward(self, x):
            """The forward pass for the model"""

            # perform the forward pass transforms
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            x = x / self.hyperdim ** 0.5  # normalize variance of output

            # return the data
            return x

    # set criterion and learning rate
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.01  # learning rate

    # make a model object on the GPU
    model = Model(hyperdim, num_classes).to(device)

    # choose the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # set model to train mode
    model.train()

    # for the number of desired passes over the data
    for epoch in tqdm(range(epochs)):

        # loop over the data in the train loader
        for inputs, labels in trainloader:

            # Put data on device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # set any weights above 1.0 to 1.0, and below -1.0 to -1.0
            with torch.no_grad():
                model.linear.weight.data = torch.clamp(
                    model.linear.weight.data, min=-1, max=1
                )

    # after the step, training passes, make all the weights in {+-1}
    with torch.no_grad():
        model.linear.weight.data = torch.sign(model.linear.weight.data)

    # return the model
    return model


def binary_majority(trainloader, num_classes, hyperdim):
    """Classifier which uses majority vote in each dimension of hyperspace"""

    # set up empty tensors for accuracy calculation
    A = torch.zeros((num_classes, hyperdim), dtype=torch.float)
    counts = torch.zeros((num_classes, 1), dtype=torch.float)

    # loop over the training data
    for i, data in enumerate(trainloader, 0):
        # loop over the number of classes
        for j in range(num_classes):

            # idx indicates which data in the batch are of class j
            idx = data[1] == j

            # copy the image data in the batch
            inputs = data[0].clone()

            # reshape the data
            inputs = inputs.view(inputs.shape[0], -1)

            # update the matrix A with the votes from input
            A[j, :] = A[j, :] + torch.sum(1 - 2 * inputs[idx, :], axis=0)

            # update the counts tensor with the votes
            counts[j] = counts[j] + torch.sum(idx)

    # transform the data in A to determine the majority vote in each hyperdimension
    A = A / counts
    A = torch.sign(A)
    A = A.to(device)

    # class to act as classifier machine
    class Model(torch.nn.Module):
        def __init__(self, A):
            super(Model, self).__init__()
            self.A = A

        def forward(self, x):
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = torch.matmul(x, self.A.T)
            return x

    # construct the classification model
    model = Model(A).to(device)

    # return the classification model
    return model


def float_majority(trainloader, num_classes, hyperdim):
    """Classifier using majority vote in float mode.

    This is largely the same as the binary_majority mode above
    """

    # set up empty tensors for accuracy calculation
    A = torch.zeros((num_classes, hyperdim), dtype=torch.float)
    counts = torch.zeros((num_classes, 1), dtype=torch.float)

    # loop over the training data
    for i, data in enumerate(trainloader, 0):
        # loop over the number of classes
        for j in range(num_classes):
            # idx indicates which data in the batch are of class j
            idx = data[1] == j

            # copy the image data in the batch
            inputs = data[0].clone()

            # reshape the data
            inputs = inputs.view(inputs.shape[0], -1)

            # update the matrix A with the votes from input
            A[j, :] = A[j, :] + torch.sum(1 - 2 * inputs[idx, :], axis=0)

            # update the counts tensor with the votes
            counts[j] = counts[j] + torch.sum(idx)

    # transform the data in A to determine the majority vote in each hyperdimension\
    # the only difference between binary mode is that we don't take the sign,
    # so this ends up just being the mean vote in each hyperdimension.
    A = A / counts
    A = A.to(device)

    # class to act as classifier machine
    class Model(torch.nn.Module):
        def __init__(self, A):
            super(Model, self).__init__()
            self.A = A

        def forward(self, x):
            x = 1 - 2 * x.type(torch.float)
            x = x.view(x.size(0), -1)
            x = torch.matmul(x, self.A.T)
            return x

    # construct the classification model
    model = Model(A).to(device)

    # return the classification model
    return model


def cnn_sgd(trainloader, num_classes, epochs, sz):
    """Classifier using convolutional neural network.

    Keyword arguments:
    trainloader: the training data
    num_classes: number of classes to classify
    epochs:      number of times to run over the data
    sz:          size of one side of the (square) input data
    """

    # make two copies of sz
    n1, n2 = sz

    class BasicCNN(nn.Module):
        """Class for the CNN classifier"""

        def __init__(self, n1, n2):
            """Initializer for the CNN classifier

            Class attributes:
            N: hyperdimensionality of the output data
            conv: convolutional neural network
            pool: max pooling transformation
            linear: linear classifier
            """

            # run the initializer for the parent class
            super(BasicCNN, self).__init__()

            # fix some model parameters
            kernel_size = 3
            stride = 1
            padding = 0
            out_channels = 16
            pool_kernel_size = 2
            pool_stride = 2

            # calculate the necessary dimensions of the output data
            m1 = (n1 - kernel_size + 2 * padding) // stride + 1
            m2 = (n2 - kernel_size + 2 * padding) // stride + 1
            k1 = (m1 - pool_kernel_size) // pool_stride + 1
            k2 = (m2 - pool_kernel_size) // pool_stride + 1
            N = k1 * k2 * out_channels

            # set they hyperdimensionality
            self.N = N

            # make the convolutional layer
            self.conv = nn.Conv2d(
                in_channels=1,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )

            # make the pooling layer
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

            # make the linear layer
            self.linear = nn.Linear(in_features=N, out_features=10)

            # set these dimensions
            self.n1 = n1
            self.n2 = n2

        def forward(self, x):
            """forward pass for the CNN classifier"""

            # reshape the data and transform to be +-1
            x = x.view(-1, 1, self.n1, self.n2)
            x = (1 - 2 * x).type(torch.float)

            # perform the convolution and pooling
            x = self.conv(x)
            x = self.pool(x)

            # reshape again
            x = x.view(x.size(0), -1)

            # apply the linear layer and normalize
            x = self.linear(x) / self.N ** 0.5

            # return teh data
            return x

    # set the classification criteria
    criterion = torch.nn.CrossEntropyLoss()

    # set the learning rate and build the classifier with the model defined above
    alpha = 0.01
    model = BasicCNN(n1, n2).to(device)

    # select an optmizer
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)

    # debug
    # print(epochs)

    # set model to training mode
    model.train()

    # for the number of epochs selected for the training
    for epoch in tqdm(range(epochs)):

        # loop over the data in the trainloader
        for i, data in enumerate(trainloader, 0):

            # copy the data and labels to the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # reset the gradients
            optimizer.zero_grad()

            # pass over the model
            outputs = model(inputs)

            # find the loss according to the selected criterion
            loss = criterion(outputs, labels)

            # perform the backwards and forwards step of the training
            loss.backward()
            optimizer.step()

    # return the model
    return model
