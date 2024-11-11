# import the necessary libraries
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
import torch.nn as nn

# set the device for extended computations to be the GPU, unless it is not available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dataset(dataname, selected_labels, data_folder, num_train=None, num_test=None):
    """Loads the desired dataset.

    Keyword arguments:
    dataname:        string literal, either "MNIST" or "FashionMNIST"
    selected_labels: tuple or list containing the set of classes you want to use
    data_folder:     string literal, relative filepath to data
    num_train:       int, # of datapoints to return for training data (default: None - use all data)
    num_tes:         int, # of datapoints to return for testing data (default: None - use all data)
    """

    # convert the selected labels to a numpy array
    selected_labels = np.array(selected_labels)
    num_classes = len(selected_labels)

    # make sure the requested data is valid
    assert dataname in ["MNIST", "FashionMNIST"], "ERROR: Dataname is not valid!"

    if dataname == "MNIST":
        # MNIST has classes that are just digits
        class_names = np.array(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

        # load the MNIST training data
        trainmnist = datasets.MNIST(
            data_folder,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Load the MNIST testing data
        testmnist = datasets.MNIST(
            data_folder,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    if dataname == "FashionMNIST":
        # The classes for FashionMNIST are types of clothes
        class_names = np.array(
            [
                "T-shirt/top",
                "Trouser",
                "Pullover",
                "Dress",
                "Coat",
                "Sandal",
                "Shirt",
                "Sneaker",
                "Bag",
                "Ankle boot",
            ]
        )

        # load the FashionMNIST training data
        trainmnist = datasets.FashionMNIST(
            data_folder,
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )

        # Load the FashionMNIST testing data
        testmnist = datasets.FashionMNIST(
            data_folder,
            train=False,
            download=True,
            transform=transforms.ToTensor(),
        )

    # take a subset of the training data if desired
    if num_train:
        trainmnist = torch.utils.data.Subset(trainmnist, range(num_train))

    # take a subset of the testing data if desired
    if num_test:
        testmnist = torch.utils.data.Subset(testmnist, range(num_test))

    # if fewer than all classes are desired, take only the datapoints with the desired classes
    if num_classes < 10:
        trainmnist = filter_dataset(trainmnist, selected_labels)
        testmnist = filter_dataset(testmnist, selected_labels)

    # set up array for the labels that will be in the returned dataset
    selected_names = class_names[selected_labels]

    # return the data as a triple (testdata, traindata, names)
    return trainmnist, testmnist, selected_names


def filter_dataset(dataset, labels, sort_data=False):
    """Filter the dataset for the selected labels.

    Keyword arguments:
    dataset:   the data to filter.
    labels:    the set of labels to select.
    sort_data: bool to sort the data so datapoints with the same label
               are continguous in the returned array
    """

    # start with an empty array
    filtered_data = []

    # loop over the dataset
    for i in range(len(dataset)):
        # when the label of that datapoint is one of the selected ones
        if dataset[i][1] in labels:
            # append this element to the filtered dataset list
            index = int(np.where(labels == dataset[i][1])[0])
            filtered_data.append([dataset[i][0], index])

    # if sorting is desired, perform the sort
    if sort_data:
        x = np.zeros(len(filtered_data), dtype=int)
        for i in range(len(filtered_data)):
            x[i] = filtered_data[i][1]
        idx = np.argsort(x)
        sorted_data = []
        for i in range(len(filtered_data)):
            sorted_data.append(filtered_data[idx[i]])
        filtered_data = sorted_data

    # return the filtered data
    return filtered_data


def plot_images(filename, images, class_names, vmin=0.0, vmax=1.0):
    """Plot a collection of images on the same figure.

    Keyword arguments:
    filename: string literal, filepath to save figure on disk.
    images: array of image data to plot.
    class_names: array of class names associated with each image at the same index.
    vmin: minimum value to get mapped to 0.0 during normalization (default: 0.0)
    vmax: maximum value to get mapped to 1.0 during normalization (default: 1.0)

    NOTE: the function expects 10 images to be in `images`, but it will work regardless.
          The purpose is to have exactly 1 image per class.
    """

    # set up figure to be displayed
    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))

    # normalize with the desired vmin and vmax
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="gray", norm=norm)
    sm.set_array([])

    # display the 10 images in a tableau together
    for i in range(10):
        img = images[i]
        row = i // 5
        col = i % 5
        axs[row, col].imshow(img.squeeze(), cmap="gray", vmin=vmin, vmax=vmax)
        axs[row, col].set_title(f"{class_names[i]}", fontsize=14)

    # formatting
    fig.colorbar(sm, ax=axs, orientation="vertical", fraction=0.02, pad=0.04)

    # save the image at the specified location
    plt.savefig(filename, format="eps")

    # show the image
    plt.show()


def plot_weights(filename, conv_weights):
    """Plot the wavelets used in the convolutional case.

    Keyword arguments:
    filename: string literal, path to save the final image.
    conv_weights: array of wavelets to plot.

    NOTE: expects at least 9 wavelets in the conv_weight list.
    """
    # set up the plots
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(7, 5))

    # loop over the subplots and plot each image
    for i, ax in enumerate(axes.flat):
        img = conv_weights[i][0]
        im = ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
        ax.set_xticks(np.arange(-0.5, img.shape[1], 1), minor=False)
        ax.set_yticks(np.arange(-0.5, img.shape[0], 1), minor=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # do some plot formatting
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    # Increase fontsize for colorbar tick labels
    cbar.ax.tick_params(labelsize=16)

    # save the image at the desired filepath
    plt.savefig(filename, format="eps")

    # show the plot
    plt.show()


def haar_convolution(n1, n2, stride):
    """Create and return the neural net for performing convolution with wavelet.

    Keyword arguments:
    n1: the first dimension of the input images
    n2: the second dimension of the input images

    """

    # set up some parameters
    kernel_size = 4
    out_channels = 9
    padding = 0

    # calculate dimensions of the result convolved data
    m1 = (n1 - kernel_size + 2 * padding) / stride + 1
    m2 = (n2 - kernel_size + 2 * padding) / stride + 1

    # find the new hyperdimensionality of the data
    N = int(m1) * int(m2) * out_channels

    # set up the convolver for the data
    haar_conv = conv = nn.Conv2d(
        in_channels=1,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
    )

    # define the haar wavelets
    haar_conv.weight.data[:, :, :, :] = 0
    haar_conv.weight.data[0, :, :, :] = 1

    haar_conv.weight.data[1, :, 2:, :] = 1
    haar_conv.weight.data[1, :, :2, :] = -1

    haar_conv.weight.data[2, :, :, 2:] = 1
    haar_conv.weight.data[2, :, :, :2] = -1

    haar_conv.weight.data[3, :, 1:2, :] = 1
    haar_conv.weight.data[3, :, :1, :] = -1

    haar_conv.weight.data[4, :, 2:3, :] = 1
    haar_conv.weight.data[4, :, 3:, :] = -1

    haar_conv.weight.data[5, :, :, 1:2] = 1
    haar_conv.weight.data[5, :, :, :1] = -1

    haar_conv.weight.data[6, :, :, 2:3] = 1
    haar_conv.weight.data[6, :, :, 3:] = -1

    haar_conv.weight.data[7, :, :2, :2] = +1
    haar_conv.weight.data[7, :, :2, 2:] = -1

    haar_conv.weight.data[8, :, :2, :2] = +1
    haar_conv.weight.data[8, :, 2:, :2] = -1

    # return the convolver and new dimensionality
    return haar_conv, N


def haar_features(dataset, haar_conv, outdim):
    """Generate the convolution feautres

    Keyword arguments:
    dataset: the data to convolve
    haar_conv:
    """

    # number of datapoints
    num_data = len(dataset.dataset)

    # empty array of appropriate size for the features
    conv_features = torch.zeros((num_data, outdim), device="cpu")

    # empty array for the labels associated with each datapoint
    conv_labels = torch.zeros(num_data, device="cpu")

    # create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # send the convolution net to the GPU (if available)
    haar_conv = haar_conv.to(device)

    # track the index we are currently at.
    i = 0

    # gradient doesn't matter for this step
    with torch.no_grad():
        # loop over the data in the dataloader
        for data, labels in data_loader:
            # Transfering images and labels to GPU if available
            num = data.shape[0]
            data, labels = data.to(device), labels.to(device)

            # convolve the data
            features = haar_conv(data)

            # print(data.shape)
            # print(features.shape)
            # sys.exit(1)

            # store the convolutional features in the output array(s)
            conv_features[i : i + num, :] = features.reshape(num, -1)
            conv_labels[i : i + num] = labels

            # update the index
            i = i + num

    # return the convolutional features and labels
    return conv_features, conv_labels

def svd_features(X):
    """
    Generate the singular value decomposition (svd) features

    INPUT:
    
    X: Input Data
    
    OUTPUT:
    
    svd_data: The input features transformed into svd features
    """
    if not isinstance(X, torch.Tensor):
        raise ValueError("X must be a torch Tensor")
    
    x_dims = len(X.shape)
    if x_dims < 2:
        raise ValueError("Dimension Error: The input array X must be at least 2-dimensional.")

    if x_dims >= 2:
        X = X.reshape(X.shape[0], -1)

    X = X.to(torch.float)/255
    U, s, V = torch.linalg.svd(X, full_matrices = False)
    B = U * s.reshape(1, -1)

    X_new = B @ V

    max_val = torch.max(X_new)
    min_val = torch.min(X_new)

    svd_data = ((X_new - min_val)/(max_val - min_val))
    
    return svd_data
