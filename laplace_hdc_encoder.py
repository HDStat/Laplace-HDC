# import the necessary libraries
import numpy as np
import torch
from torch.utils.data import TensorDataset
import sys

# Use the GPU if avaiable, else use the CPU for larger computations
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class hdc_encoder:
    """Class for encoding input data as hypervectors

    The encoder is defined the dimensionality of input data,
    the dimensionality of the output hypervectors,
    the color similarity kernel, and the encoding mode it uses.
    The size of the color similarity kernel, K, will have an entry
    for each pair of possible colors. For example, in a grayscale image
    with 256 colors, K would be a 256x256 tensor.

    To create an encoder, use
    hdc_encoder(inputdim, hyperdim, K, mode)

    Attributes:
    inputdim: int, dimensionality of the input data
    hyperdim: int, number of hyperdimensions
    K: 2d tensor, the color similarity kernel.
    random_permutations: the family of permutations to use (when applicable)
    mode: string, the encoding mode to use.
    """

    def __init__(self, inputdim, hyperdim, K, mode):
        """Initializer for"""
        # set up class variables
        # the encode_batch and sz are yet to be determined
        self.inputdim = inputdim
        self.hyperdim = hyperdim
        self.K = K
        self.mode = mode
        self.encode_batch = None
        self.sz = None

        # set of random permutations can start empty
        # since we won't always use this
        self.random_permutations = None

        # check that the mode is valid
        assert self.mode in [
            "shift_1d",
            "shift_2d",
            "block_diag_shift_1d",
            "block_diag_shift_2d",
            "rand_permutation",
        ]

        # set up the variables depending on the selected mode
        if self.mode == "shift_1d":
            self.encode_batch = self.shift_1d
            self.sz = (1, self.hyperdim)

        elif self.mode == "shift_2d":
            self.encode_batch = self.shift_2d
            n = int(np.sqrt(self.hyperdim))
            self.hyperdim = n ** 2
            self.sz = (n, n)

        elif self.mode == "block_diag_shift_1d":
            self.encode_batch = self.block_diag_shift_1d
            n = self.hyperdim // self.inputdim
            self.hyperdim = n * self.inputdim
            self.sz = (n, self.inputdim)

        elif self.mode == "block_diag_shift_2d":
            self.encode_batch = self.block_diag_shift_2d
            n1 = int(np.sqrt(self.inputdim))
            n2 = self.hyperdim // self.inputdim
            assert n1 ** 2 == self.inputdim
            self.hyperdim = n2 * n1 ** 2
            self.sz = (n2, n1, n1)

        # if the random permutatio mode is selected...
        elif self.mode == "rand_permutation":
            # randomly sample permutations and stack them into a tensor
            # Each row is a permutation for a hypervector,
            # and there is a row for each pixel position in the input image
            self.random_permutations = torch.stack(
                [torch.randperm(self.hyperdim) for _ in range(self.inputdim)]
            )
            self.encode_batch = self.random_permutation
            self.sz = (1, self.hyperdim)

        # this should never be triggered due to the assertion above,
        # but this is good to have anyways just in case.
        else:
            print("hdc_encoder: invalid mode")
            sys.exit(1)

        # The following method follows the embedding method described in the paper.

        # num is the number of colors in the image
        num = self.K.shape[0]

        # generate W and find its eigendecomposition
        W = np.sin(np.pi / 2.0 * self.K)
        eigen_values, eigen_vectors = np.linalg.eigh(W)

        # force U to be positive definite by getting rid of
        # negative or 0 eigenvalues
        U = np.sqrt((np.maximum(0, eigen_values))).reshape(-1, 1) * eigen_vectors.T
        
        # convert to float
        U = torch.from_numpy(U).to(torch.float).to(device)

        # build V, the matrix with hypervectors for each pixel color
        G = torch.randn(self.hyperdim, num).to(device)
        V = torch.sign(G @ U)

        # convert to binary based on the signs of the entries
        self.V = ((V >= 0).T).type(torch.bool).to(device)

        return

    def encode(self, data_loader):
        """Encode the image.

        The data_loader has data stored in batches to begin with.

        Keyword arguments:
        data_loader: the data loader with the data to be encoded.
        """

        # n is the number of images in the dataset
        n = len(data_loader.dataset)

        # empty tensors to store the encoded hypervectors and labels
        Ux = torch.zeros((n, self.hyperdim), dtype=torch.bool)
        labels = torch.zeros(n).long()

        # number of colors the data may contain (typically 256 for MNIST data)
        num_colors = self.K.shape[0]

        # index to insert the hypervectors in Ux
        i0 = 0

        # loop over the batches in the dataloader
        for batch in data_loader:
            # get the number of images in the given batch.
            # this isn't a constant since it could be
            # smaller near the end of the data_loader
            num_imgs = batch[0].size(0)

            # find the ending index for where to insert the data
            i1 = i0 + num_imgs

            # ??? ask nick about this one
            batch_data = (
                ((num_colors - 1) * batch[0].reshape(num_imgs, 1, -1)).type(torch.long)
            ).to(device)

            # encode the data and place into the Ux matrix
            Ux[i0:i1] = self.encode_batch(batch_data)

            # update the labels
            labels[i0:i1] = batch[1]

            # update the new starting index
            i0 = i1

        Ux = Ux.reshape((Ux.shape[0],) + self.sz)

        # wrap the Ux and labels into a dataset and return
        return TensorDataset(Ux, labels)

    def shift_1d(self, x):
        """Simple 1D shifting method"""

        # Constants for brevity
        N = self.hyperdim
        d = self.inputdim

        # Encode each data as hypervector
        Y = (self.V[x.flatten()].reshape(-1, d, N)).to(device)
        Y = Y.permute(1, 2, 0)

        # Resulting hypervector
        U = torch.zeros(Y.shape[1:], dtype=torch.bool).to(device)

        # Bitwise XOR
        for i in range(d):
            U[:i, :] = torch.bitwise_xor(U[:i, :], Y[i, N - i :, :])
            U[i:, :] = torch.bitwise_xor(U[i:, :], Y[i, : N - i, :])

        # Permute and reshape
        U = U.permute(1, 0).reshape(-1, N)

        return U

    def shift_2d(self, x):
        """Simple 2D cyclic shifting method"""

        # Constants
        N = self.hyperdim
        d = self.inputdim
        dim = int(np.sqrt(N))
        side = int(np.sqrt(d))

        # Encode each data as hypervector
        Y = (self.V[x.flatten()].reshape(-1, d, N)).to(device)
        Y = Y.permute(1, 2, 0)
        Y = Y.reshape(side, side, dim, dim, -1)

        # Resulting hypervector
        U = torch.zeros(Y.shape[2:], dtype=torch.bool).to(device)

        # Bitwise XOR
        for i in range(side):
            for j in range(side):
                U[i:, j:, :] = torch.bitwise_xor(
                    U[i:, j:, :], Y[i, j, : dim - i, : dim - j, :]
                )
                U[:i, :j, :] = torch.bitwise_xor(
                    U[:i, :j, :], Y[i, j, dim - i :, dim - j :, :]
                )
                U[i:, :j, :] = torch.bitwise_xor(
                    U[i:, :j, :], Y[i, j, : dim - i, dim - j :, :]
                )
                U[:i, j:, :] = torch.bitwise_xor(
                    U[:i, j:, :], Y[i, j, dim - i :, : dim - j, :]
                )

        # Permute and reshape
        U = U.permute(2, 0, 1).reshape(-1, N)

        return U

    def block_diag_shift_1d(self, x):
        """Block diagonal 1D shifting method"""

        # Constants
        N = self.hyperdim
        d = self.inputdim
        Np = N // d

        # Encode each data as hypervector
        Y = (self.V[x.flatten()].reshape(-1, d, N)).to(device)
        Y = Y.permute(1, 2, 0)
        Y = Y.reshape(d, d, Np, -1)

        # Resulting hypervector
        U = torch.zeros(Y.shape[1:], dtype=torch.bool).to(device)

        for i in range(d):
            U[:i, :, :] = torch.bitwise_xor(U[:i, :, :], Y[i, d - i :, :, :])
            U[i:, :, :] = torch.bitwise_xor(U[i:, :, :], Y[i, : d - i, :, :])

        # Permute and reshape
        U = U.permute(2, 1, 0).reshape(-1, N)

        return U

    def block_diag_shift_2d(self, x):
        """Block diagonal 2D shifting method"""

        # Constants
        N = self.hyperdim
        d = self.inputdim
        Np = N // d
        side = int(np.sqrt(d))

        # Encode each data as hypervector
        Y = (self.V[x.flatten()].reshape(-1, d, N)).to(device)
        Y = Y.permute(1, 2, 0)
        Y = Y.reshape(side, side, side, side, Np, -1)

        # Resulting hypervector
        U = torch.zeros(Y.shape[2:], dtype=torch.bool).to(device)

        # Bitwise XOR
        for i in range(side):
            for j in range(side):
                U[i:, j:, :, :] = torch.bitwise_xor(
                    U[i:, j:, :, :], Y[i, j, : side - i, : side - j, :, :]
                )
                U[:i, :j, :, :] = torch.bitwise_xor(
                    U[:i, :j, :, :], Y[i, j, side - i :, side - j :, :, :]
                )
                U[i:, :j, :, :] = torch.bitwise_xor(
                    U[i:, :j, :, :], Y[i, j, : side - i, side - j :, :, :]
                )
                U[:i, j:, :, :] = torch.bitwise_xor(
                    U[:i, j:, :, :], Y[i, j, side - i :, : side - j, :, :]
                )

        # Permute and reshape
        U = U.permute(3, 2, 0, 1).reshape(-1, N)

        return U

    def random_permutation(self, x):
        """Random permutations

        Given all the pixels in the image, each pixel location
        gets permuted by a uniformly selected permutation.
        """

        # Constants
        N = self.hyperdim
        d = self.inputdim

        # Encode each data as hypervector
        Y = (self.V[x.flatten()].reshape(-1, d, N)).to(device)
        Y = Y.permute(1, 2, 0)

        # loop over all the pixels
        for i in range(d):
            Y[i, :, :] = Y[i, self.random_permutations[i], :]

        # U will be the resulting hypervector
        U = torch.zeros(Y.shape[1:], dtype=torch.bool).to(device)

        #
        for j in range(Y.shape[2]):
            for i in range(d):
                U[:, j] = torch.bitwise_xor(U[:, j], Y[i, :, j])

        # Permute and reshape
        U = U.permute(1, 0).reshape(-1, N)

        return U
