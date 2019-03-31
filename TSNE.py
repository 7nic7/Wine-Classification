from sklearn import manifold
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def prepare(file = 'C:/Users/tianping/Desktop/winequality-red.csv'):

    origin_data = pd.read_csv(file)
    # print(data.columns)
    data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
    data = np.array(data, dtype='float32')
    data[:, :-1] = preprocessing.minmax_scale(data[:, :-1], axis=0)

    return pd.DataFrame(data, columns=origin_data.columns[0].split(';'))


data = prepare()
data_n = np.array(data)
tsne = manifold.TSNE()
two = tsne.fit_transform(data_n[:, :-1])

labels = np.zeros(data_n.shape[0])
for i in range(data_n.shape[0]):
    if data_n[i, -1] in [1, 2, 3, 8, 9, 10]:
        labels[i] = 1

plt.scatter(two[labels==1, 0], two[labels==1, 1], label='anomaly', c='r')
plt.scatter(two[labels==0, 0], two[labels==0, 1], label='normal', c='k', alpha=0.3)
plt.legend()
plt.show()



