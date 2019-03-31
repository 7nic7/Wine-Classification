import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense,Input
from sklearn import preprocessing
from keras.losses import binary_crossentropy
from keras.optimizers import adadelta
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc



def prepare(file = 'C:/Users/tianping/Desktop/winequality-red.csv'):

    origin_data = pd.read_csv(file)
    # print(data.columns)
    data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
    data = np.array(data, dtype='float32')
    data[:, :-1] = preprocessing.minmax_scale(data[:, :-1], axis=0)

    return pd.DataFrame(data, columns=origin_data.columns[0].split(';'))

def split_train_val_test(data, train_ratio=0.6, shuffle=False,
                         good_class=[4, 5, 6, 7], bad_class=[1, 2, 3, 8, 9, 10]):


    good_example = np.concatenate([data[index,:] for index,class_ in enumerate(data[:,-1])\
                                   if class_ in good_class]).reshape([-1, 12])
    good_example_nums = good_example.shape[0]
    bad_example = np.concatenate([data[index,:] for index,class_ in enumerate(data[:,-1]) \
                                  if class_ in bad_class]).reshape([-1, 12])
    bad_example_nums = bad_example.shape[0]

    if shuffle:

        index1 = list(range(good_example_nums))
        np.random.shuffle(index1)
        good_example = good_example[index1]
        index2 = list(range(bad_example_nums))
        np.random.shuffle(index2)
        bad_example = bad_example[index2]

    # train 选择完数据之后， 剩下的对半分
    total_example_nums = data.shape[0]
    train_num = int(np.floor(total_example_nums * train_ratio))
    assert good_example_nums > train_num
    val_good_num = int((good_example_nums - train_num) // 2)
    val_bad_num = int(bad_example_nums // 2)

    train_X = good_example[:train_num, :-1]
    train_y = good_example[:train_num, -1]

    val_X_good = good_example[train_num:(train_num + val_good_num), :-1]
    val_X_bad = bad_example[:val_bad_num, :-1]
    val_X = np.concatenate([val_X_good,val_X_bad]).reshape([-1, 11])
    val_y_good = good_example[train_num:(train_num + val_good_num), -1]
    val_y_bad = bad_example[:val_bad_num, -1]
    val_y = np.concatenate([val_y_good,val_y_bad])

    test_X_good = good_example[(train_num + val_good_num):, :-1]
    test_X_bad = bad_example[val_bad_num:, :-1]
    test_X = np.concatenate([test_X_good, test_X_bad]).reshape([-1, 11])
    test_y_good = good_example[(train_num + val_good_num):, -1]
    test_y_bad = bad_example[val_bad_num:, -1]
    test_y = np.concatenate([test_y_good, test_y_bad])

    return train_X, train_y, val_X, val_y, test_X, test_y


d = prepare()
data_n = np.array(d)
train_X, train_y, val_X, val_y, test_X, test_y = split_train_val_test(data_n)

train_num = train_X.shape[0]
val_num = val_X.shape[0]


encoding_dim = 2  # 80 floats -> compression of factor 0.8, assuming the input is 100 floats
# hidden_dim = 8
# this is our input placeholder
input = Input(shape=(11,))
# "encoded" is the encoded representation of the input
# hidden = Dense(hidden_dim, activation='relu')(input)
encoded = Dense(encoding_dim, activation='relu')(input)
# hidden1 = Dense(hidden_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(11, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input, outputs=decoded)

# this model maps an input to its encoded representation
encoder = Model(inputs=input, outputs=encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
# decoder_hidden_layer = autoencoder.layers[-2](encoded_input)
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(inputs=encoded_input, outputs=decoder_layer(encoded_input))


autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


# encoded = encoder.predict(data_n.values)
# decoded = decoder.predict(encoded)
# naivedist = np.zeros(len(data_n.values))
# for i, x in enumerate(data_n.values):
#     naivedist[i] = np.linalg.norm(x-decoded[i])



autoencoder.fit(train_X, train_X,
                epochs=2500,
                batch_size=100,
                shuffle=True,
                verbose=1)


encoded = encoder.predict(val_X)
decoded = decoder.predict(encoded)


dist = np.zeros(shape=[val_num, 1])
for i, x in enumerate(val_X):
    dist[i] = np.linalg.norm(x-decoded[i]) # euclidean distance

labels = np.zeros(shape=[val_num, 1])

for i,class_ in enumerate(val_y):
    if class_ in [1,2,3,8,9,10]:
        labels[i] = 1


fpr, tpr, thresholds = roc_curve(labels, dist)
roc_auc = auc(fpr, tpr)



plt.figure(figsize=(10,6))
plt.plot(fpr, tpr, color='red', label='AUC = %0.2f)' % roc_auc)
plt.xlim((0,1))
plt.ylim((0,1))
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.title('ROC Autoencoder 100-80-100 ReLU/Sigmoid synth\_multidim\_100\_000', fontsize=16)
plt.legend(loc="lower right")
plt.show()
# plt.savefig("ae-outlier-training-roc.svg", format="svg")


# plt.figure(figsize=(10, 7))
# plt.scatter(np.arange(0, train_num + val_num), dist, c=labels, edgecolor='black', s=15)
# plt.xlabel('Index')
# plt.ylabel('Score')
# plt.xlim((0, train_num + val_num))
# plt.title("Outlier Score")
# plt.show()
# plt.savefig("ae-outlier-training.svg", format="svg")


# def compute_error_per_dim(point):
#     p = np.array(data_n[point,:-1]).reshape([1, 11])
#     encoded = encoder.predict(p)
#     decoded = decoder.predict(encoded)
#     return np.array(p - decoded)[0]
#
# plt.figure(figsize=(10, 7))
# plt.plot(np.arange(1,12), compute_error_per_dim(-1))
# plt.plot(np.arange(1,12), compute_error_per_dim(-2))
# plt.plot(np.arange(1,12), compute_error_per_dim(-3))
# plt.xlim((1, 11))
# plt.xlabel('Index')
# plt.ylabel('Reconstruction error')
# plt.title("Reconstruction error in each dimension of point ??")
