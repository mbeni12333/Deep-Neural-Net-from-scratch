import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn
plt.style.use('seaborn')
def initialize_parameters(layer_dims, debug=False):
    """
    This function initializes each layers weights and bayes
        and other params in the future, and returns a dictionary

    :param layer_dims: a list of numbers describing the number of units
                        in every layer
    :return: a dictionary containing all the paramaters for each layer
    """
    nb_layers = len(layer_dims)
    params = {}

    for i in range(1, nb_layers):
        # He initilization
        he = np.sqrt(2/layer_dims[i-1])
        # params W and b
        params["W"+str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1])*he
        params["b"+str(i)] = np.zeros((layer_dims[i], 1))

    # return the dictionary
    return params

def random_permutation(X, Y, minibatch_size=128, debug=False):
    """
    This function takes for parameter some data and it returnes a list of
        minibatches (x, y) randomly permuted

    :param X: np array of shape (nbFeatures, m_training exemple)
    :param Y: np array of shape (1, m_training exemple)
    :param minibatch_size: well .. the size of a minibatch
    :param debug: to check later if the function is working as expected
    :return: minibatches (a list of batches randomply permuted collumns)

    """
    minibatches = []    # this will be returned
    nb_exemple = X.shape[1]     # the number of training exemples
    nb_minibatches = nb_exemple // minibatch_size
    if debug:
        print(f"nb_minibatches = {nb_minibatches}")

    permutation = list(np.random.permutation(nb_exemple)) # we create a permutation matrix
    # then we shuffle our data with the same permutation
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    for i in range(0, nb_minibatches):
        # we slice the minibatches one at a time
        minibatch_X = shuffled_X[:, i * minibatch_size:(i + 1) * minibatch_size]
        minibatch_Y = shuffled_Y[:, i * minibatch_size:(i + 1) * minibatch_size]
        #
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append(minibatch)
    # if the last batch is smaller then the size of the mini batch
    # create a minibatch of that size
    if nb_exemple %  minibatch_size != 0:

        nb_minibatches_passed =   int(math.floor(nb_exemple / minibatch_size))
        # the collumn to start from
        end = minibatch_size*nb_minibatches_passed
        if debug:
            print(f"end = {end}")
        # we slice it
        minibatch_X = shuffled_X[:, end:]
        minibatch_Y = shuffled_Y[:, end:]

        # and we zip it
        minibatch = (minibatch_X, minibatch_Y)
        minibatches.append((minibatch))
    # just in case we want to check later
    if debug:
        s=0
        for minibatchx, minibatchy in minibatches:
            s+=minibatchx.shape[1]
        print(f"s = {s}, x.shape[1] = {x.shape[1]}")
        assert(s == shuffled_X.shape[1])

    # return the full list of minibatches
    return minibatches


def sigmoid(Z):
    """
    This function calculates the sigmoid of Z

    :param Z: np array of any size
    :return: A, activation_cache
    """
    # calculate the sigmoid
    A = 1./(1+np.exp(-Z))
    # we will need that later in the backward phase
    activation_cache = Z

    # return the thing
    return A, activation_cache

def sigmoid_backward(dA, cache):
    """
    This function return the derivative of the sigmoid during the backward phase

    :param dA: gradient of the next A
    :param cache: activation_cache
    :return: dZ
    """
    Z = cache
    s = 1./(1+np.exp(-Z))

    # the derivative of the activation with respect to Z * dA of the
    # next layer (chain rule) to get the derivative of the loss with respectof z

    dZ = dA * s * (1-s)

    # return the derivative

    return dZ

def relu(Z):
    """
    This function calculates the relu of the input

    :param Z: np array of any size
    :return:the A , activation_cache
    """
    A = np.maximum(0, Z)

    activation_cache = Z

    # return the stuff
    return A, activation_cache

def relu_backward(dA, cache):
    """
    This function calculates the derivative of the activation with respect of z

    :param A: the activation of the current layer
    :param cache: the activation cache of the current layer
    :return: the derivative of the activation with respect to Z
    """
    # derivative of relu is 1 if z > 0 and 0 otherwise
    # so to get the derivative of the loss with respect of z
    # we need to multiply the current dA (chain rule)

    dZ = np.array(dA, copy=True) #so we basicly just copy dA

    # we don't forget to make the derivative 0 as well when it's 0
    dZ[cache < 0] = 0

    # return the derivative

    return dZ

def linear_backward(dZ, cache):
    """
    This function calculate the derivative of the linear activation with respect
        to A_prev, W, b ..

    :param dZ: gradient of the linear activation with respect to Z
    :param cache: linear cache (A_prev, w, b)
    :return: dA_prev, dW, db
    """

    # unpack the linear cahe
    A_prev, W, b  = cache
    m = A_prev.shape[1]
    # calculate the derivatives
    dW = 1./m*dZ.dot(A_prev.T)

    db = 1./m*np.sum(dZ, axis=1, keepdims=True)

    dA_prev = W.T.dot(dZ)
    # debug

    assert(dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    # return the hole thing
    return dA_prev, dW, db
if __name__ == "__main__":

    np.random.seed(0)
    kwargs = dict(histtype='stepfilled', alpha=0.3, normed=True, bins=50)

    # test case for random permutation
    #x = np.random.randn(4,1300)
    #y = np.random.randn(1,1300)

    #mini = random_permutation(x, y, debug=True)

    # test case for initialization
    #param = initialize_parameters([786,100,100])
    #for k,v in param.items():
    #    print(f"key : {k}, value : \n{v}")
    #    if k[0] != 'b':
    #        plt.hist(v.reshape(v.shape[0]*v.shape[1], 1), **kwargs)
    #plt.legend()
    #plt.show()


    #linear backward test cases
    b = np.random.randn(5, 1)
    w = np.random.randn(5,4)
    A = np.random.randn(4,100)
    dZ = np.random.randn(5,100)

    cache = (A, w, b)

    dA, dW, db = linear_backward(dZ, cache)
    print(f"dA = \n{dA}, \ndW = \n{dW},\ndb = \n{db}")