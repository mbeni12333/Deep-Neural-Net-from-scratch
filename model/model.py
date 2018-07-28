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
                AL, caches = self.model_forward(minibatch_X)
                ###compute cost
                cost = self.compute_cost(AL, minibatch_Y)
                if self.debug:
                    utills.log(cost)
                ###backward
                self.grads = self.model_backward(AL, minibatch_Y)
                ###update
                self.parameters = self.update_parameters()
        return




    def model_forward(self, X):


    def model_backward(self, AL, Y):


    def linear_activation_forward(A_prev, W, b, activation="relu"):


    def linear_activation_backward(dA, cache, activation):


    def gradient_checking(self):
        return

    def update_parameters(self):
        ## gradient
        return
