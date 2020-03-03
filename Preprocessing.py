import pandas as pd
import numpy as np

def load_data():
    file = open('Data/pendigits.tra')
    data = file.readlines()
    data = np.array([np.array(x.replace('\n','').split(',')) for x in data])
    data = data.astype(float)
    train_x, train_y = data[:, :16], data[:, 16].reshape(data.shape[0],1)

    file = open('Data/pendigits.tes')
    data = file.readlines()
    data = np.array([np.array(x.replace('\n', '').split(',')) for x in data])
    data = data.astype(float)
    test_x, test_y = data[:, :16], data[:,16].reshape(data.shape[0],1)

    return train_x.T, train_y.T, test_x.T, test_y.T
