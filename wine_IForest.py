from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt


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

data = prepare()
data_n = np.array(data)
X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(data_n)

labels_test = np.zeros(y_test.shape[0])
for index, i in enumerate(y_test):
    if i in [1, 2, 3, 8, 9, 10]:
        labels_test[index] = 1

labels_val = np.zeros(y_val.shape[0])
for index, i in enumerate(y_val):
    if i in [1, 2, 3, 8, 9, 10]:
        labels_val[index] = 1

# fit the model

param = {}
for max_features_ in range(1, 12):
    for contamination_ in np.arange(0, 0.2, 0.01):
        clf = IsolationForest(n_estimators=300, contamination=contamination_, max_samples=32, bootstrap=False,
                              max_features=max_features_)
        clf.fit(X_train)

        val_score = clf.decision_function(X_val)

        x, y, threshold = roc_curve(labels_val, -val_score)
        a = auc(x, y)
        print('max_features:%s  contamination:%s  auc:%s' % (max_features_, contamination_, a))
        param[a] = (max_features_, contamination_)

best_model = list(param.keys())[np.argmax(list(param.keys()))]
best_param = param[best_model]
print('best:auc:%s   contamination:%s   max_features:%s'%(best_model, best_param[1], best_param[0]))

clf = IsolationForest(n_estimators=300, contamination= best_param[1], max_samples=256, bootstrap=False,
                      max_features=best_param[0])
clf.fit(X_train)

test_score = clf.decision_function(X_test)

x, y, threshold = roc_curve(labels_test, -test_score)
a = auc(x, y)

plt.plot(x, y, c='red', label='%s auc' % a)
plt.plot([0, 1], [0, 1], c='navy', linestyle='--')
plt.legend()
plt.show()
plt.title('IForest')
