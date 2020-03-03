import numpy as np
import pandas as pd
import math

# we initialize 3 layer
def initialization(input_dim, hidden_dim, output_dim):

    W1 = np.random.randn(hidden_dim,input_dim) * 0.01
    b1 = np.zeros((hidden_dim, 1))
    W2 = np.random.randn(output_dim, hidden_dim) * 0.01
    b2 = np.zeros((output_dim, 1))

    initialized = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return initialized

def feed_forward(X, initialized):
    W1 = initialized['W1']
    b1 = initialized['b1']
    W2 = initialized['W2']
    b2 = initialized['b2']

    # layer 1
    Z1 = np.dot(W1, X) + b1
    A1 = 1/(1 + np.exp(-Z1))
    # layer 2
    Z2 = np.dot(W2, A1) + b2
    A2 = 1/(1 + np.exp(-Z2))
    A2 = np.exp(Z2)
    # softmax_layer
    denominator = 1/np.sum(A2, axis=1)
    A2 = A2*denominator.reshape(10,1)
    y_pred = np.zeros_like(A2)
    for i in range(A2.shape[1]):
        result = np.where(A2[:,i] == np.amax(A2[:,i]))
        if len(result[0]) < 1:
            y_pred[0,i] = 1
        else:
            y_pred[result[0][0], i] = 1

    caches = {'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2}
    params = {'W1': W1, 'W2':W2, 'b1':b1, 'b2':b2}

    return y_pred, caches, params

def compute_loss_MSE(Y, A2):
    L = 1/2 *((Y - A2)**2)
    return L

def compute_loss(Y, A2):
    m = Y.shape[1]
    L = -1/m*np.sum(np.multiply(Y, np.log(A2)))
    return L

def gradiant_decent(X, Y, caches, params, learning_rate):
    m = Y.shape[1]
    Z1 = caches['Z1']
    A1 = caches['A1']
    Z2 = caches['Z2']
    A2 = caches['A2']

    W1 = params['W1']
    W2 = params['W2']
    b1 = params['b1']
    b2 = params['b2']

    # dA2 = Y - A2      # for the MSE
    dA2 = -Y/A2
    dZ2 = np.multiply(dA2, A2)/m
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)*(1/m)
    dZ1 = np.dot(W2.T, dZ2) * A1*(1 - A1)
    dW1 = np.dot(dZ1, X.T)
    db1= np.sum(dZ1, axis=1, keepdims=True)*(1/m)

    W1 = W1 - learning_rate*dW1
    W2 = W2 - learning_rate*dW2
    b1 = b1 - learning_rate*db1
    b2 = b2 - learning_rate*db2

    new_params = {'W1':W1, 'W2':W2, 'b1':b1, 'b2':b2}

    return new_params

def compute_accuracy(y_pred, Y):
    total = Y.shape[1]
    cnt = 0
    for i in range(total):
        x = np.dot(y_pred[:,i].T, Y[:,i])
        if x == 1:
            cnt += 1
    acc = cnt/total
    return acc
