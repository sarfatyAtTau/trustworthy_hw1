import gzip
import numpy as np
import consts

class Model:

    def __init__(self, layers, lr, loss):
        """
        Initialize the model layers, and set the learning rate
        and loss. The loss can be one of binary cross-entropy (bce),
        categorical cross-entropy (cce), or mean-squared error (mse).
        """
        self.layers = layers
        self.lr = lr
        self.loss = loss.lower()
        assert(loss=='bce' or loss=='cce' or loss=='mse')

    def predict(self, inputs):
        """
        Perform the forward pass on a batch of inputs, and return
        the outputs of the last layer (batch_size X n_classes).
        """
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_dldi(self, x, y, clear_state=True, loss=None):
        """
        Compute the gradient of the loss w.r.t. the input. If 
        clear_state is set to True (default), clear each layer's
        state after computing the backward pass, otherwise keep
        the state (e.g., to update layer parameters). If loss is
        not provided, use the built-in model loss (i.e., self.loss).
        """
        pass

    def fit_step(self, x_train, y_train):
        """
        Given a batch of samples and their corresponding labels, 
        compute the gradients w.r.t. layer parameters, and perform
        the SGD update (on a sinlge batch).
        """
        pass

    def save_weights(self, fpath):
        """
        Saves the weights at a given path
        """
        with gzip.open(fpath, 'wb') as fout:
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    np.save(fout, layer.W)
                if hasattr(layer, 'b'):
                    np.save(fout, layer.b)
                if hasattr(layer, 'T'):
                    np.save(fout, layer.T)

    def load_weights(self, fpath):
        """
        Loads the weights from a given path
        """
        with gzip.open(fpath, 'rb') as fin:
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    layer.W = np.load(fin)
                if hasattr(layer, 'b'):
                    layer.b = np.load(fin)
                if hasattr(layer, 'T'):
                    layer.T = np.load(fin)

class Layer:

    def __init__(self):
        self.input = 0

    def forward(self, inputs, store=False):
        """
        Computes the forward pass of a layer. If store is set 
        to True, then store the necessary state to perform a
        backward computation.
        """
        pass

    def backward(self, dldo):
        """
        Receives the gradient of the loss w.r.t. the layer output,
        and returns the gradient of the loss w.r.t. the layer input.
        If there are parameters that require updating, then this 
        method also computes the gradients w.r.t. these parameters.
        """
        pass

    def update_params(self, lr):
        """
        Updates the parameters with a step-size / learning-rate 
        of lr.
        """
        pass

    def clear_state(self):
        """
        Clear the internal state stored in a previous forward pass
        """
        pass

class DenseLayer(Layer):
    """
    A fully connected layer, with input_dim inputs, and
    output_dim outputs
    """
    def __init__(self, input_dim, output_dim, init_sigma=0.05):
        super.__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # rand init
        self.W = np.random.randn(input_dim, output_dim)*init_sigma
        self.b = np.random.randn(1, output_dim)*init_sigma

        self.dW = 0
        self.db = 0

    def forward(self, inputs, store=False):
        x = np.add(np.matmul(self.W, inputs), self.b)
        if store:
            self.input = inputs

    def backward(self, dldo):
        # calculate grads for back prop
        dodi = np.matmul(self.W.T, dldo)

        # internal updates
        # TODO: average over batch
        self.dW = np.matmul(dldo.reshape(-1, 1), self.input.reshape(1, -1))
        self.db = dldo
        return dodi

    def update_params(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

    def clear_state(self):
        pass


class SigmoidActLayer(Layer):
    """
    The Sigmoid activation function
    """
    def forward(self, inputs, store=False):
        pass

    def backward(self, dldo):
        pass

    def clear_state(self):
        pass


class ReLUActLayer(Layer):
    """
    The ReLU activation function (ReLU(x)=max(0, x)).
    """

    def __init__(self):
        super.__init__()

    def forward(self, inputs, store=False):
        if store:
            self.input = inputs
        # TODO: ReLU in Numpy
        return np.max(0, inputs)

    def backward(self, dldo):
        # TODO: ReLU in Numpy
        return np.sign(np.max(0, self.input))

    def clear_state(self):
        pass

class SoftmaxActLayer(Layer):
    """
    The softmax activation function, with temperature T
    (set to 1, by default).
    """
    def __init__(self, T=1):
        self.T = T

    def forward(self, inputs, store=False):
        pass

    def backward(self, dldo):
        pass

    def clear_state(self):
        pass
