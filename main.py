import Preprocessing as pp
import NeuralNet as nn
import numpy as np

train_x, train_y, test_x, test_y = pp.load_data()
train_y = train_y.T
train_y = train_y.astype(int)
Y = np.zeros((train_y.shape[0], 10))
for i in range(train_y.shape[0]):
    Y[i][train_y[i][0]] = 1
Y = Y.T

test_y = test_y.T
test_y = test_y.astype(int)
Y_test = np.zeros((test_y.shape[0], 10))
for i in range(test_y.shape[0]):
    Y_test[i][test_y[i][0]] = 1
Y_test = Y_test.T

m = train_x.shape[1]
n_x = train_x.shape[0]
n_y = 10
n_h = 1024
epoches = 16
learning_rate = 1E-4
batch = 100

initialized = nn.initialization(n_x, n_h, n_y)

for i in range(epoches):
    l, itr = 0, int(m/batch)
    print('Epoach ', str(i))
    for j in range(itr):
        if (l+batch) > m:
            X = train_x[:,l:]
            _Y = Y[:,l:]
        else:
            X = train_x[:, l:l+batch]
            _Y = Y[:, l:l+batch]
        y_pred, caches, params = nn.feed_forward(X, initialized)
        loss = nn.compute_loss(_Y, caches['A2'])
        print('-'*10, 'Batch number ',str(j), ' with Loss: ', str(loss), ' accuracy is: ', nn.compute_accuracy(y_pred, _Y)*100)
        initialized = nn.gradiant_decent(X, _Y, caches, params, learning_rate)
        l += batch

# now it is time fot testing
y_pred, caches, params = nn.feed_forward(test_x, initialized)

print('Accuracy of the model on the test dataset is: ', nn.compute_accuracy(y_pred, Y_test)*100,'%')
