import consts
import numpy as np

def eval_acc(model, x, y):
    """
    Evaluate the accuracy of the model, given a set
    of samples and their corresponding ground-truth 
    labels.
    """
    preds = []
    for i in range(len(x)):
        preds.append( model.predict(x[i].reshape((-1,))) )
    preds = np.array(preds)
    return np.sum((preds==y)/len(y))

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

class BinaryLogisticRegression:

    def __init__(self, input_dim, lr, init_sigma=0.01):
        """
        Receive the input dimensionality, the learning rate, and 
        the std of the zero-centered normal distribution, to init
        the weight matrix and bias.
        """
        self.input_dim = input_dim
        self.lr = lr
        # rand init
        self.W = np.random.randn(input_dim)*init_sigma
        self.b = np.random.randn()*init_sigma

    def predict_proba(self, x):
        """
        Estimate the probability of class 1 using the sigmoid
        function
        """
        linear_result = np.add(np.matmul(self.W, x), self.b)
        return 1/(1 + np.exp(-linear_result))

    def predict(self, x):
        """
        Returns the most likely class 0 or 1 (using a threshold of 0.5).
        """
        if self.predict_proba(x) < 0.5:
            return 0
        else:
            return 1

    def fit_step(self, x, y):
        """
        Given a pair (x,y), a sample and its ground truth label,
        update the weights vector W and the bias parameter b, and
        return the binary cross-entropy loss.
        """
        y_hat = self.predict_proba(x)
        loss = -(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
        # calculated by chain rule
        dw = x*(y_hat - y)
        db = y_hat - y

        self.W = self.W - self.lr*dw
        self.b = self.b - self.lr*db
        return loss

class MulticlassLogisticRegression:

    def __init__(self, input_dim, lr, n_classes, init_sigma=0.01):
        """
        Receive the input dimensionality, learning rate, number of 
        classes, and  std of the zero-centered normal distribution, 
        to init the weight matrix and bias.
        """
        self.input_dim = input_dim
        self.lr = lr
        self.n_classes = n_classes
        # rand init
        self.W = np.random.randn(n_classes, input_dim)*init_sigma
        self.b = np.random.randn(n_classes)*init_sigma

    def predict_proba(self, x):
        """
        Estimate the probability of each class using the softmax 
        function
        """
        s = np.squeeze(np.matmul(self.W, x[:, np.newaxis])) + self.b
        return softmax(s)

    def predict(self, x):
        """
        Returns the top class
        """
        return np.argmax(self.predict_proba(x))

    def fit_step(self, x, y):
        """
        Given a pair (x,y), a sample and its ground truth label,
        update the weight matrix W and the bias vector b, and
        return the cross-entropy loss.
        """
        y_hat = self.predict_proba(x)
        loss = -np.log(y_hat[y])
        # calculated by chain rule
        y_one_hot = np.zeros(self.n_classes)
        y_one_hot[y] = 1
        dw = np.matmul((y_hat - y_one_hot).reshape(-1, 1), x.reshape(1, -1))
        db = y_hat - y_one_hot

        self.W = self.W - self.lr * dw
        self.b = self.b - self.lr * db
        return loss
