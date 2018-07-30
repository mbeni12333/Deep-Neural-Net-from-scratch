from utills import *
import os
import copy
import sklearn.datasets as sk
from sklearn.model_selection import train_test_split

class model(object):
    """
        This model creates a L layer Neural Network

    """

    def __init__(self, X, Y, layers_dims=[1], learning_rate=0.1,
                 beta1=0.9, beta2=0.99,batch_size=64, epoch=100, lambd=0,
                 epsilon=1e-7, keep_fact=1, debug=True,
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
        self.parameters ,self.V, self.S = initialize_parameters(layers_dims)
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
        self.logf = os.getcwd()+'/logs'





    def train(self, learning_rate):
        k=0
        params = self.parameters
        v = self.V
        s = self.S
        pa = []
        grads  = {}
        costs = []
        for i in range(1, self.nb_epoch):

            ### randomly permutate examples in batch
            minibatches = random_permutation(self.X, self.Y, minibatch_size=self.batch_size)
            p = self.nb_epoch
            for j, (minibatch_X, minibatch_Y) in enumerate(minibatches):

                ### forward propagation
                AL, caches = self.model_forward(minibatch_X, params)

                ###compute cost
                cost = compute_cost(AL, minibatch_Y, params, self.nb_layer, self.lambd)
                costs.append(cost)
                ###backward
                grads = self.model_backward(AL, minibatch_Y, caches, self.lambd)
                #if self.debug:
                #    for i in grads.keys():
                #        print(f"{i}")
                ###update
                params, self.V, self.S = self.update_parameters(params,self.V, self.S, grads,learning_rate, self.beta1, self.beta2)
                plot_decision_boundary(lambda x: simple_model.predict_dec(parameters=params, X=x.T), self.X, self.Y, k, self.logf)
                k = k+1
            if self.debug:
                k = int(i/p*100/2)
                t = "="*k
                s = '-'*(50-k)
                space = ' '*int((np.log(p)/np.log(10) - np.log(i+1)/np.log(10)))
                print(f"{i}/{self.nb_epoch} {space} [{t}>{s}] , cost = {cost}")
            #print(f"pa = {pa}")


            self.parameters = params
        return costs




    def model_forward(self, X, params):
        """

        :param X: The mini batch matrix
        :param parameters: dictionary
        :return: AL, caches
        """

        L = self.nb_layer
        caches = []
        A = X
        for i in range(1,L-1):
            # variables
            A_prev = A
            W = params["W"+str(i)]
            b = params["b"+str(i)]

            # the forward step
            A, cache = linear_activation_forward(A_prev, W, b, 'relu')
            # store curernt cache to use it later
            caches.append(cache)

        # variables
        WL = params["W"+str(L-1)]
        bL = params["b"+str(L-1)]

        # forward step
        AL, cache = linear_activation_forward(A, WL, bL, 'sigmoid')
        caches.append(cache)

        return AL, caches
    def model_backward(self, AL, Y, caches, lambd=0.1):
        """
        This function calculate gradient using backpropagation algorithm

        :param AL: last layer
        :param Y: expected output
        :param caches: list of caches
        :return: grads
        """
        L = len(caches)
        #print(f"L is {L}")
        grads = {}

        #initialize backprop
        dAL = -(np.divide(Y, AL) -  np.divide(1-Y, 1-AL))
        current_cache = caches[L-1]

        dA, dW, db = linear_activation_backward(dAL, current_cache, 'sigmoid', lambd)

        grads['dA'+str(L-1)] = dA
        grads['dW'+str(L)] = dW
        grads["db"+str(L)] = db

        for i in reversed(range(L-1)):

            current_cache = caches[i]
            #print(f"Current cache : {i}")
            dA, dW, db = linear_activation_backward(dA, current_cache, 'relu')

            grads['dA' + str(i)] = dA
            grads['dW' + str(i+1)] = dW
            grads["db" + str(i+1)] = db

        return grads

    def gradient_checking(self):
        return

    def update_parameters(self, params, V, S, grads, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-7):
        """
        This function uses gradient descent algorithm to compute the next
            step gradient
        :param grads: dictionary containing the gradiant of every parameter
        :param layer_dims: list of number of units per layer
        :param learning_rate: learning  rate
        :return: the new params dictionary
        """
        L = self.nb_layer
        v_corrected = {}
        s_corrected = {}

        for i in range(1, L):
            V["W"+str(i)] = V["W"+str(i)]*beta1 + (1-beta1)*grads["dW"+str(i)]
            V["b"+str(i)] = V["b"+str(i)]*beta1 + (1-beta1)*grads["db"+str(i)]

            v_corrected["W"+str(i)] = V["W"+str(i)] / np.sqrt(1/np.square(beta1+epsilon)+epsilon)
            v_corrected["b"+str(i)] = V["b"+str(i)] / np.sqrt(1/np.square(beta1+epsilon)+epsilon)

            S["W"+str(i)] = S["W"+str(i)]*beta2 + (1-beta2)*np.square(grads["dW"+str(i)])
            S["b"+str(i)] = S["b"+str(i)]*beta2 + (1-beta2)*np.square(grads["db"+str(i)])

            s_corrected["W"+str(i)] = S["W"+str(i)] / np.sqrt(1/np.square(beta2+epsilon)+epsilon)
            s_corrected["b"+str(i)] = S["b"+str(i)] / np.sqrt(1/np.square(beta2+epsilon)+epsilon)

            # update step
            params["W"+str(i)] -= learning_rate*v_corrected["W"+str(i)]/np.sqrt(s_corrected["W"+str(i)]+epsilon)
            params["b"+str(i)] -= learning_rate*v_corrected["b"+str(i)]/np.sqrt(s_corrected["b"+str(i)]+epsilon)


        return params, v_corrected, s_corrected

    def predict(self, X,Y,  parameters):

        m = X.shape[1]
        n = len(parameters) // 2
        p = np.zeros((1, m))

        # Forward propagation
        probas, caches = self.model_forward(X, parameters)


        for i in range(0, probas.shape[1]):
            if probas[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0


        print("Accuracy: " + str(np.sum((p == Y) / m)))

        return p

    def predict_dec(self, parameters, X):
        """
        Used for plotting decision boundary.

        Arguments:
        parameters -- python dictionary containing your parameters
        X -- input data of size (m, K)

        Returns
        predictions -- vector of predictions of our model (red: 0 / blue: 1)
        """

        # Predict using forward propagation and a classification threshold of 0.5
        a3, cache = self.model_forward(X, parameters)
        predictions = (a3 > 0.5)
        return predictions



if __name__ == '__main__':

   X, Y = sk.make_moons(n_samples=4000, noise=.1)
   X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
   X_train = X_train.T
   Y_train = Y_train.reshape(Y_train.shape[0], 1).T
   X_test = X_test.T
   Y_test = Y_test.reshape(Y_test.shape[0], 1).T

   f = plt.figure()
   f = f.gca()
   print(f"train shape X : {X_train.shape}, Y : {Y_train.shape}")
   np.random.seed(0)
   simple_model = model(X_train, Y_train, [20,20,20,1],batch_size=256, epoch=10, lambd=0.01, beta1=0.9, beta2=0.995)
   input('press enter')
   costs= simple_model.train(learning_rate=0.01)
   _ = simple_model.predict(X_test, Y_test, simple_model.parameters)
   f.plot(np.arange(0, len(costs), 1), np.array(costs))

   #print(f"is it ?  : {params[5] == params[2]}")


   #a = anim.FuncAnimation(ff2, animate, frames=2)
   #a.save('animation.mp4', fps=30)
   plt.show()
