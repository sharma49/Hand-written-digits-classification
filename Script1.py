import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import random
import pickle

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W

def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return (1.0 / (1.0 + np.exp(-1.0 * z)))

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('./mnist_all.mat') #loads the MAT object as a Dictionary

    #Your code here
    unsample_train_data = np.array([0]*784)
    unsample_train_label = np.array([0])
    test_data = np.array([0]*784)
    test_label = np.array([0])

    for i in range(10):
        testx = mat.get('test'+str(i))
        testx = testx.astype('float32')
        testx = testx/255
        test_data = np.vstack((test_data, testx))
        label = np.array([i])
        label = np.array([label]*testx.shape[0])
        test_label = np.vstack((test_label,label))

    test_data = np.delete(test_data,(0),axis=0)
    test_label = np.delete(test_label,(0),axis=0)

    for i in range(10):
        trainx = mat.get('train'+str(i))
        trainx = trainx.astype('float32')
        trainx = trainx/255
        unsample_train_data = np.vstack((unsample_train_data, trainx))
        label = np.array([i])
        label = np.array([label]*trainx.shape[0])
        unsample_train_label = np.vstack((unsample_train_label,label))

    unsample_train_data = np.delete(unsample_train_data,(0),axis=0)
    unsample_train_label = np.delete(unsample_train_label,(0),axis=0)

    sample_train = random.sample(range(60000),50000)
    sample_validation = list(set(range(60000)) - set(sample_train))

    train_data = unsample_train_data[np.array(sample_train)]
    train_label = unsample_train_label[np.array(sample_train)]
    validation_data = unsample_train_data[np.array(sample_validation)]
    validation_label = unsample_train_label[np.array(sample_validation)]

    # Feature Selection Start
    complete_data=np.array(np.vstack((train_data, validation_data, test_data)))
    rejected_features = np.all(complete_data == complete_data[0,:], axis = 0)
    complete_data = complete_data[:,~rejected_features]

    train_data = complete_data[0:len(train_data),:]
    validation_data = complete_data[len(train_data): (len(train_data) + len(validation_data)),:]
    test_data = complete_data[(len(train_data) + len(validation_data)): (len(train_data) + len(validation_data) + len(test_data)),:]
    # Feature Selection End

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    #Feed Forward Begins
    # Adding input bias node to Training Data
    input_bias_node = np.zeros(len(training_data))
    input_bias_node.fill(1)
    training_data = np.column_stack([training_data,input_bias_node])

    w1_T = np.transpose(w1)
    aj = np.dot(training_data,w1_T)
    # Output from hidden node
    zj = sigmoid(aj)
    # Adding hidden bias node to zj
    zj_bias_node = np.zeros(len(training_data))
    zj_bias_node.fill(1)
    zj = np.column_stack([zj,zj_bias_node])
    # Calculating output form output node
    w2_T = np.transpose(w2)
    bl = np.dot(zj,w2_T)
    ol = sigmoid(bl)
    #Feed Forward End

    # Creating a label vector yl from training label
    yl = np.zeros((len(training_data),10))
    for l in range(len(training_data)):
        yl[l][training_label[l]] = 1

    # Gradiance Calculation Begin
    delta_l = (ol - yl) * (1 - ol) * ol
    # Gradiance of EF wrt wt from hidden to output node (w2)
    grad_w2 = np.dot(np.transpose(delta_l),zj)

    grad_w1_p1 = (1 - zj) * zj
    grad_w1_p2 = np.dot(delta_l,w2)
    grad_w1_p3 = grad_w1_p1 * grad_w1_p2
    # Gradiance of EF wrt wt from input to hidden node (w1)
    grad_w1 = np.dot(np.transpose(grad_w1_p3),training_data)
    grad_w1 = grad_w1[0:n_hidden,:]
    # Gradiance End

    # Error Function Begin
    error_function = np.sum((0.5 * ((yl - ol)**2)))/(len(training_data))
    # Error Function End

    # Regularization Begin
    # Gradiance of EF wrt w1 and w2 after adding Regularization coeff
    grad_w1 = (grad_w1 + (lambdaval*w1))/len(training_data)
    grad_w2 = (grad_w2 + (lambdaval*w2))/len(training_data)
    # New EF after adding Regularization term
    reg_func = error_function + (lambdaval/(2*len(training_data)))*(np.sum(w1*w1) + np.sum(w2*w2))
    obj_val = reg_func
    print (obj_val)
    #Regularization End

    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    return (obj_val,obj_grad)

def nnPredict(w1,w2,data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in hidden
    %     layer to unit j in output layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.zeros((data.shape[0],1))
    #Your code here

    # Feed Forward Begins
    input_bias_node = np.zeros(len(data))
    input_bias_node.fill(1)
    data = np.column_stack([data,input_bias_node])
    w1_T = np.transpose(w1)
    aj = np.dot(data,w1_T)
    # Output from hidden node
    zj = sigmoid(aj)
    # Adding Hidden Bias node to zj
    hidden_bias_node = np.zeros(len(data))
    hidden_bias_node.fill(1)
    zj = np.column_stack([zj,hidden_bias_node])
    # Calculating output of output node
    w2_T = np.transpose(w2)
    bl = np.dot(zj,w2_T)
    ol = sigmoid(bl)
    # Feed Forward End

    # Prediction Begin
    for x in range(ol.shape[0]):
        max_index = np.argmax(ol[x])
        labels[x] = max_index

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1];

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0.6;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

pickle.dump([n_hidden, w1, w2, lambdaval], open("params1.pickle", "wb"))

#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')