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
        x = inputs.reshape(-1, 1, inputs.shape[-1])
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

        pred = x.reshape(-1, 1, x.shape[-1])
        for layer in self.layers:
            pred = layer.forward(pred, store=True)
        if loss is None:
            loss = self.loss

        assert pred.shape[0] == y.shape[0], "Batch dimension problem"
        if loss == 'mse':
            # loss = (pred - y)^2
            dl_dpred = 2 * (pred - y.reshape(y.shape[0], 1, -1))

        elif loss == 'cce':
            # loss = -sum_over_i(y_i * log(pred_i))
            # dl_dpred_i = -y_i/pred_i
            dl_dpred = np.zeros((y.shape[0], 1, pred.shape[-1]))
            for sample in range(y.shape[0]):
                dl_dpred[sample][0][y[sample]] = - 1/pred[sample][0][y[sample]]
        else:
            assert False, "Should not get here"

        dldo = np.transpose(dl_dpred, axes=[0, 2, 1])
        for layer in reversed(self.layers):
            dldo = layer.backward(dldo)
            if clear_state:
                layer.clear_state()
        return dldo

    def fit_step(self, x_train, y_train):
        """
        Given a batch of samples and their corresponding labels, 
        compute the gradients w.r.t. layer parameters, and perform
        the SGD update (on a sinlge batch).
        """
        self.compute_dldi(x_train, y_train, clear_state=False)
        for layer in self.layers:
            layer.update_params(self.lr)

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
        self.state = 0

    def forward(self, inputs, store=False):
        """
        Computes the forward pass of a layer. If store is set 
        to True, then store the necessary state to perform a
        backward computation.
        """
        if store:
            self.state = inputs

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
        self.state = 0

class DenseLayer(Layer):
    """
    A fully connected layer, with input_dim inputs, and
    output_dim outputs
    """
    def __init__(self, input_dim, output_dim, init_sigma=0.05):
        super(DenseLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # rand init
        self.W = np.random.randn(input_dim, output_dim)*init_sigma
        self.b = np.random.randn(1, output_dim)*init_sigma

        self.dW = 0
        self.db = 0

    def forward(self, inputs, store=False):
        assert not np.isnan(inputs).any()
        super(DenseLayer, self).forward(inputs, store)
        assert len(inputs.shape) == 3 and inputs.shape[1] == 1 and inputs.shape[2] == self.input_dim, "Dimesions problem"
        return np.add(np.matmul(inputs, self.W), self.b)


    def backward(self, dldo):
        # calculate grads for back prop
        assert len(dldo.shape) == 3 and dldo.shape[1] == self.output_dim and dldo.shape[2] == 1, "Dimesions problem"
        dldi = np.matmul(self.W, dldo)

        # internal updates
        self.dW = np.mean(np.matmul(np.transpose(self.state, axes=[0, 2, 1]), np.transpose(dldo, axes=[0, 2, 1])), axis=0)
        self.db = np.mean(np.transpose(dldo, axes=[0, 2, 1]), axis=0)

        assert len(dldi.shape) == 3 and dldi.shape[1] == self.input_dim and dldi.shape[2] == 1, "Dimesions problem"
        return dldi

    def update_params(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db

    def clear_state(self):
        super(DenseLayer, self).clear_state()


class SigmoidActLayer(Layer):
    """
    The Sigmoid activation function
    """

    def sigmoid_func(self, x):
        return 1/(1 + np.exp(-x))

    def forward(self, inputs, store=False):
        sigmoid = self.sigmoid_func(inputs)
        super(SigmoidActLayer, self).forward(sigmoid, store)
        return sigmoid

    def backward(self, dldo):
        assert len(dldo.shape) == 3 and len(self.state.shape) == 3, "Dimesions problem"
        assert dldo.shape[0] == self.state.shape[0] and dldo.shape[1] == self.state.shape[2] and dldo.shape[2] == self.state.shape[1], "Dimesions problem"
        grad = np.multiply(self.state, 1-self.state)
        return np.multiply(dldo, np.transpose(grad, axes=[0, 2, 1]))

    def clear_state(self):
        super(SigmoidActLayer, self).clear_state()


class ReLUActLayer(Layer):
    """
    The ReLU activation function (ReLU(x)=max(0, x)).
    """

    def __init__(self):
        super(ReLUActLayer, self).__init__()

    def forward(self, inputs, store=False):
        super(ReLUActLayer, self).forward(inputs, store)
        return np.maximum(0, inputs)

    def backward(self, dldo):
        return np.multiply(np.transpose(np.sign(np.maximum(0, self.state)), axes=[0, 2, 1]), dldo)

    def clear_state(self):
        super(ReLUActLayer, self).clear_state()

class SoftmaxActLayer(Layer):
    """
    The softmax activation function, with temperature T
    (set to 1, by default).
    """
    def __init__(self, T=1):
        super(SoftmaxActLayer, self).__init__()
        self.T = T

    def forward(self, inputs, store=False):
        exp_inputs = np.exp(inputs)
        sum_of_exps = np.sum(exp_inputs, axis=2, keepdims=True)
        output = exp_inputs/sum_of_exps
        if store:
            self.state = output
        return output

    def backward(self, dldo):
        assert len(dldo.shape) == 3 and len(self.state.shape) == 3, "Dimesions problem"
        assert dldo.shape[0] == self.state.shape[0] and dldo.shape[1] == self.state.shape[2] and dldo.shape[2] == self.state.shape[1], "Dimesions problem"
        s_diag = np.zeros((self.state.shape[0], self.state.shape[-1], self.state.shape[-1]))
        for sample in range(self.state.shape[0]):
            s_diag[sample] = np.diag(self.state[sample].flatten())
        dsdi = s_diag - np.matmul(np.transpose(self.state, axes=[0, 2, 1]), self.state)
        dldi = np.matmul(dsdi, dldo)
        return dldi

    def clear_state(self):
        self.state = 0
