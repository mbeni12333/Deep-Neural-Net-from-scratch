from .utills import *

class model(object):
    """
        This model creates a L layer Neural Network

    """

    def __init__(self, X, Y, layers_dims=[1], learning_rate=0.1,
                 beta1=0.9, beta2=0.99,batch_size=64, epoch=3000, lambd=0,
                 epsilon=1e-7, keep_fact=1, debug=False,
                 log_file='', load_weights=''):
        """
        This function initializes all attributes of this model

        :param X: training data
        :param Y: training data labels
        :param layers_dims: list of number unit par layer
        :param learning_rate: rate of learning during gradient descent
        :param beta1: momentum, rmsprop factor
        :param beta2: adam factor
        :param batch_size: size of bacthe
        :param epoch: number of training episodes
        :param lambd: L2 regularization
        :param epsilon: small number to avoid dividing by zero
        :param keep_fact: dropout regularization , list of percentage of keep for each layer
        :param debug: debug
        :param log_file: where to save logs
        :param load_weights: path to save model to
        """


        # architecture
        self.layer_dims = layers_dims.insert(0, X.shape[0])
        self.nb_layer = len(layers_dims)
        self.batch_size = batch_size
        self.nb_epoch = epoch
        # learnable parameters
        self.parameters = {}
        self.V = {}
        self.S = {}
        # static hyperparameters
        self.learning_rate = learning_rate
        self.keep_prob = keep_fact
        self.lambd = lambd
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        # data set
        self.X = X
        self.Y = Y
        # some info
        self.nb_exemples = self.X.shape[1]
        self.debug = debug






    def train(self):

        params = self.parameters
        grads  = {}
        costs = []
        for i in range(1, self.nb_epoch):

            ### randomly permutate examples in batch
            minibatches = random_permutation(self.X, self.Y)
            p = len(minibatches)
            for j, (minibatch_X, minibatch_Y) in enumerate(minibatches):

                ### forward propagation
                AL, caches = self.model_forward(minibatch_X, params)

                ###compute cost
                cost = compute_cost(AL, minibatch_Y, params, self.nb_layer)
                if self.debug:
                    k = int(j/p*100)
                    t = "="*k
                    s = '-'*(100-k)
                    space = ' '*int((np.log(p)/np.log(10) - np.log(j)/np.log(10)))
                    print(f"{j}/{p} {space} [{t}{s}] , cost = {cost}")
                ###backward
                grads = self.model_backward(AL, minibatch_Y)
                ###update
                self.parameters = self.update_parameters()
        return




    def model_forward(self, X, params):
        """

        :param X: The mini batch matrix
        :param parameters: dictionary
        :return: AL, caches
        """

        L = self.nb_layer
        caches = []

        for i in range(1,L):

            line

    def model_backward(self, AL, Y, caches):
        return

    def gradient_checking(self):
        return

    def update_parameters(self, params, grads,learning_rate):
        """
        This function uses gradient descent algorithm to compute the next
            step gradient
        :param grads: dictionary containing the gradiant of every parameter
        :param layer_dims: list of number of units per layer
        :param learning_rate: learning  rate
        :return: the new params dictionary
        """
        L = self.nb_layer

        for i in range(1, L):
            # update step
            params["W"+str(i)] -= learning_rate*grads["dW"+str(i)]
            params["b"+str(i)] -= learning_rate*grads["db"+str(i)]


        return params
