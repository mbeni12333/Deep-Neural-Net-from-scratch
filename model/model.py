import math
import numpy as np

from . import utills


class model(object):
    """
        put ome tet in here
    """

    def __init__(self, X, Y, layers_dims=[1], learning_rate=0.1,
                 beta1=0.9, beta2=0.99, learning_rate_decay=1,
                 batch_size=64, epoch=3000, lambd=0,
                 epsilon=1e-7, keep_fact=1, debug=False,
                 log_file='', load_weights=''):
        self.learning_rate = learning_rate
        self.layer_dims = layers_dims.insert(0, X.shape[0])
        self.nb_layer = len(layers_dims)
        self.batch_size = batch_size
        self.nb_minibatches = int(math.floor(X.shape[1] / self.batch_size))
        self.parameters = self.initialize_parameters(layers_dims)
        self.nb_epoch = epoch
        self.lambd = lambd
        self.epsilon = epsilon
        self.exp_avg_grad, self.exp_avg_squared_grad = self.initialize_Adam()
        self.X = X
        self.Y = Y
        self.caches = []
        self.grads = {}
        self.nb_exemples = self.X.shape[1]
        self.debug = debug
        self.keep_prob = keep_fact

    def initialize_Adam(self):

        v = {}
        s = {}
        for i in range(len(self.layer_dims)):
            v["W" + str(i)] = np.zeros_like(self.parameters["W" + str(i)])
            v["b" + str(i)] = np.zeros_like(self.parameters["b" + str(i)])
            v["u" + str(i)] = np.zeros_like(self.parameters["u" + str(i)])
            v["g" + str(i)] = np.zeros_like(self.parameters["g" + str(i)])
            s["W" + str(i)] = np.zeros_like(self.parameters["W" + str(i)])
            s["b" + str(i)] = np.zeros_like(self.parameters["b" + str(i)])
            s["u" + str(i)] = np.zeros_like(self.parameters["u" + str(i)])
            s["g" + str(i)] = np.zeros_like(self.parameters["g" + str(i)])
        return v, s

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        L2Reg = self.lambd / m * \
                (np.sum(np.square(np.array([self.parameters["W" + str(i)] for i in range(1, self.nbLayers)]))))
        cost = -1 / m * (Y.dot(AL.T) + (1 - Y)(np.log((1 - AL).T))) + L2Reg
        cost = np.squeeze(cost)
        ### log the cost
        return cost

    def train(self):
        for i in range(1, self.nb_epoch):
            ### randomly permutate examples in batche
            minibatches = self.random_permutation(self.X, self.Y)
            for (minibatch_X, minibatch_Y) in minibatches:
                ### forward propagation
                AL, self.caches = self.model_forward(minibatch_X)
                ###compute cost
                cost = self.compute_cost(AL, minibatch_Y)
                if self.debug:
                    utills.log(cost)
                ###backward
                self.grads = self.model_backward(AL, minibatch_Y)
                ###update
                self.parameters = self.update_parameters()
        return

    def test(self, test_x, test_y):
        return

    def random_permutation(self, X, Y):
        minibatches = []
        permutation = list(np.random.permutation(self.nb_exemples))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation]
        for i in range(0, self.nb_minibatches):
            minibatch_X = shuffled_X[:, i * self.batch_size:(i + 1) * self.batch_size]
            minibatch_Y = shuffled_Y[:, i * self.batch_size:(i + 1) * self.batch_size]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append(minibatch)
        if self.nb_exemples % self.batch_size != 0:
            end = self.nb_exemples - self.batch_size * int(math.floor(self.nb_exemples / self.batch_size))
            minibatch_X = shuffled_X[:, end:]
            minibatch_Y = shuffled_Y[:, end:]
            minibatch = (minibatch_X, minibatch_Y)
            minibatches.append((minibatch))
        return minibatches

    def initialize_parameters(self, layer_dims):
        ### improve initialization
        params = {}
        L = len(layer_dims)
        for i in range(1, L):
            ### he initialization
            params["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(
                2 / self.layer_dims[i - 1])
            params["b" + str(i)] = np.zeros((layer_dims[i], 1))
        return params

    def load_model(self, file_location='/model.weights'):
        return

    def save_model(self):
        return

    def model_forward(self, X):
        A = X
        caches = []
        for i in range(self.nb_layer):
            A_prev = A
            W, b = self.parameters["W" + str(i)], self.parameters["b" + str(i)]
            A, Z = self.linear_activation_forward(A_prev, W, b, "ReLu")
            D = np.random.rand(A.shape[0], A.shape[1])
            D = D < self.keep_prob
            A = A * D
            A = A / self.keep_prob
            caches.append(Z, D, A)
        WL, bL = self.parameters["W" + str(self.nb_layer)], self.parameters["b" + str(self.nb_layer)]
        AL, Z = self.linear_activation_forward(A, WL, bL, "sigmoid")
        caches.append((Z, D, AL))
        return AL, caches

    def model_backward(self, AL, Y):
        grads = {}
        l = self.nb_layer
        dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = self.caches[l - 1]
        grads["dA" + str(l - 1)], grads["dW" + str(l - 1)], grads["db" + str(l - 1)] = self.linear_activation_backward(
            dAL,
            current_cache,
            "sigmoid")
        for i in reversed(range(l - 1)):
            current_cache = self.caches[i]
            grads["dA" + str(l)], grads["dW" + str(l + 1)], grads["db" + str(l + 1)] = self.linear_activation_backward(
                grads["dA" + str(i + 1)], current_cache, "ReLu")
        return grads

    def linear_activation_forward(self, A_prev, W, b, activation="ReLu"):
        Z = W.dot(A_prev) + b
        if activation == "ReLU":
            A = utills.relu(Z)
        if activation == "Sigmoid":
            A = utills.sigmoid(Z)
        return A, Z

    def linear_activation_backward(self, dA, cache, activation):
        linear_cache, D, activation_cache = cache
        if activation == "ReLu":
            dZ = utills.relu_backward(dA, activation_cache)
        elif activation == "sigmoid":
            dZ = utills.sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = utills.linear_backward(dZ, linear_cache)
        dW = dW * D
        return dA_prev, dW, db

    def gradient_checking(self):
        return

    def update_parameters(self):
        ## gradient
        return
