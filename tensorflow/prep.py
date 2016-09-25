import pandas as pd
import numpy as np


# preprocessing
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
rawdata = np.asarray(pd.read_csv(url, sep='\s+', header=None)) # first 13 columns of the data is X and the last one is Y
train_x = np.transpose(rawdata[:,0:13]) # each column of the x corresponds to one training example
train_y = np.reshape(rawdata[:,-1],(1,506)) # each column corresponds to one ygold

meanx = np.mean(train_x, axis=1)
meanx = np.reshape(meanx, (meanx.shape[0],1))
stdx = np.std(train_x, axis=1, ddof=1)
stdx = np.reshape(stdx, (stdx.shape[0],1))

x = np.divide((np.subtract(train_x, meanx)), stdx)
np.savez('datafile', x, train_y) # here save the normalized x and correct y values
