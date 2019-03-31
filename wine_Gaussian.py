from sklearn import manifold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from collections import Counter
import sklearn
from sklearn import ensemble
from sklearn import preprocessing
from sklearn import model_selection, metrics
from sklearn.metrics import recall_score
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from imblearn.over_sampling import SMOTE
from pprint import pprint
# train_num = 1000
# val_num = 300
# def prepare(file = 'C:/Users/tianping/Desktop/winequality-white.csv'):
#
#     origin_data = pd.read_csv(file)
#     # print(data.columns)
#     data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
#     data = np.array(data, dtype='float32')
#
#     new_class = np.zeros_like(data[:, -1], dtype='int')
#     weights = {1: 0, 2: 0, 3: 0}
#     for index, class_ in enumerate(data[:, -1]):
#         if class_ in [5, 6, 7]:
#             new_class[index] = 2
#             weights[2] += 1
#         elif class_ in [1, 2, 3, 4]:
#             new_class[index] = 1
#             weights[1] += 1
#         elif class_ in [8, 9, 10]:
#             new_class[index] = 3
#             weights[3] += 1
#
#     basic = np.max(list(weights.values()))
#     for key in weights:
#         weights[key] = basic / weights[key]
#
#     index = list(range(data.shape[0]))
#
#     while len(set(new_class[:train_num]).intersection([1, 2, 3])) != 3 or len(set(new_class[train_num:(val_num+train_num)]).intersection([1, 2, 3])) != 3:
#         print('shuffle')
#         np.random.shuffle(index)
#         data = data[index]
#         new_class = new_class[index]
#     return pd.DataFrame(data, columns=origin_data.columns[0].split(';')), new_class, weights


# d, new_y, weights = prepare()
#
# # d = d.ix[:, 3:]
# for i in range(1, 11):
#     print(i, '-------->', sum(d.ix[:, -1] == i))

# n = d.shape[0]
# np.random.shuffle(d)
# train = d.ix[range(1000), :]
# val = d.ix[range(1000, 1300), :]
# test = d.ix[range(1300, n), :]
# print('--')
# less = [1, 2, 3, 4, 8, 9, 10, 7]
# much = [5, 6]
# must_have = pd.DataFrame([train.ix[i, :] for i in range(train.shape[0]) if train.ix[i, -1] in less])
# other = pd.DataFrame([train.ix[i, :] for i in range(train.shape[0]) if train.ix[i, -1] in much])

# epochs = other.shape[0] // must_have.shape[0]
# must_have_array = np.array(must_have)
# other_array = np.array(other)
# train = np.array(train)
# val = np.array(val)


# size = must_have_array.shape[0]
# d_array = np.array(d)

# over_samples = SMOTE()
#
# n_class = len(set(new_y))
#
# X_train = d_array[:train_num, :-1]
# X_val = d_array[train_num:(train_num+val_num), :-1]
#
#
#
# y_train = new_y[:train_num]
# print(y_train)
# y_val = new_y[train_num:(train_num+val_num)]
#
# X_train_resample, y_train_resample = over_samples.fit_sample(X_train, y_train)

# print(weights)
# scores = []
# train_score = []
# kf = model_selection.KFold(n_splits=5)
# print('start training')
#
# weights[1] *= 100
# for k in range(5, 15):
#
#     print('------------------')
#     cls = ensemble.RandomForestClassifier(n_estimators=10,
#                                             min_samples_leaf=k,
#                                           ).fit(X_train_resample, y_train_resample)
#     y_pre = cls.predict(X_val)
#     print(cls.score(X_val, y_val))
#     print(recall_score(y_val, y_pre, average='macro'))
#
# plt.plot(scores, label='val')
# plt.plot(train_score, label='train')
# plt.legend()
# plt.show()

# print(d.head())
# tsne = manifold.TSNE().fit_transform(d_array[:,:-1])
#
# quality = new_y
# #, 'yellow', 'green', 'pink', 'purple', 'orange', 'navy', 'salmon'
# colors = ['red', 'blue', 'black', 'yellow', 'green', 'pink', 'purple', 'orange', 'navy', 'salmon']
#
# for i in range(1, 11):
#     # if (i in [5,6,7]):
#     index = (quality == i)
#     plt.scatter(tsne[index, 0], tsne[index, 1], c=colors[i-1], label=str(i))
#
# plt.legend()
# plt.show()
#
# print(scores)

# pca = PCA(n_components=3).fit(d_array[:, :-1])
# two = pca.fit_transform(d_array[:, :-1])
# quality = d_array[:,-1]
# colors = ['red', 'blue', 'black', 'yellow', 'green', 'pink', 'purple', 'orange', 'navy', 'salmon']
# # ax = plt.axes(projection='3d')
# for i in [3,4,7,8]:
#
#     plt.scatter(two[quality==i, 0], two[quality==i,1],c=colors[i-1], label=i)
#
# plt.legend()
# plt.show()
# for i in [1,3,4,5,9,10]:
#     d_array[:,i] = np.log(d_array[:,i])

# for i in range(11):
#     plt.figure(i)
#     plt.hist(d_array[:, i], bins=50)
#
# plt.show()


class GaussianOutlierDetection:

    def __init__(self, data, good_class, bad_class, epsilon, min_feature_nums):

        self.data = data
        self.good_class = good_class
        self.bad_class = bad_class
        self.epsilon = epsilon
        self.min_feature_nums = min_feature_nums
        self.best = 0
        self.best_features_scores = tuple()
        self.features = list(range(11))  # use all of the features

    def split_train_val_test(self, train_ratio, shuffle=False):


        good_example = np.concatenate([self.data[index,:] for index,class_ in enumerate(self.data[:,-1])\
                                       if class_ in self.good_class]).reshape([-1, 12])
        good_example_nums = good_example.shape[0]
        bad_example = np.concatenate([self.data[index,:] for index,class_ in enumerate(self.data[:,-1])\
                                       if class_ in self.bad_class]).reshape([-1, 12])
        bad_example_nums = bad_example.shape[0]

        if shuffle:

            index1 = list(range(good_example_nums))
            np.random.shuffle(index1)
            good_example = good_example[index1]
            index2 = list(range(bad_example_nums))
            np.random.shuffle(index2)
            bad_example = bad_example[index2]

        # train 选择完数据之后， 剩下的对半分
        total_example_nums = self.data.shape[0]
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


    def fit(self, features=list(range(11))):

        self.mu = np.mean(self.train_X[:, features], axis=0, keepdims=True)
        # self.sigma2 = np.var(self.train_X[:, features], axis=0, keepdims=True)
        self.covar = np.cov(self.train_X[:, features], rowvar=False)

        self.pre_val_y = self.predict(self.val_X[:, features])
        recall,precision,f1_score = self.evaluate(self.pre_val_y, self.val_y)

        return recall, precision, f1_score

    def choose_best(self, train_ratio=0.6, features=list(range(11))):

        self.ans = {}
        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y = \
            self.split_train_val_test(train_ratio=train_ratio)

        scores = self.fit()
        self.best = scores[2]
        self.ans[1] = (self.features, scores)
        self.is_break = False

        count = 1
        while len(self.features) > self.min_feature_nums:
            count += 1
            self.is_break = True
            for i in self.features:
                features = self.features.copy()
                features.remove(i)
                self.select_features(features)
            if self.is_break:
                break
            self.features = self.best_features_scores[0].copy()
            self.ans[count] = self.best_features_scores

    def select_features(self, features):
        scores = self.fit(features=features)
        if scores[2] > self.best:
            self.best = scores[2]
            self.best_features_scores = (features, scores)
            self.is_break = False

    def predict(self, X, y=None):

        # prob = np.exp(np.sum(-np.square(X - self.mu) / 2 / self.sigma2, axis=1)) *\
        #        (1/np.sqrt(2 * np.pi)) ** X.shape[1]

        prob = (1/np.sqrt(2 * np.pi)) ** X.shape[1] / np.sqrt(np.linalg.det(self.covar)) *\
            np.exp(-0.5 * np.diag(np.dot(np.dot((X - self.mu), np.linalg.inv(self.covar)), (X - self.mu).T)))
        assert prob.size == X.shape[0]
        pre_val_y = np.zeros_like(prob)
        pre_val_y[prob < self.epsilon] = 1  # outlier

        self.prob = prob

        return pre_val_y

    def evaluate(self, prediction, truth):

        truth_v2 = np.zeros_like(truth)
        index1 = [i for i in range(truth.size) if truth[i] in self.good_class]
        index2 = [i for i in range(truth.size) if truth[i] in self.bad_class]
        truth_v2[index1] = 0
        truth_v2[index2] = 1

        recall = np.sum((truth_v2 + prediction) == 2) / np.sum(truth_v2 == 1)
        precision = np.sum((truth_v2 + prediction) == 2) / np.sum(prediction == 1)
        f1_score = 2 * precision * recall / (precision + recall)

        self.truth_v2 = truth_v2

        return recall, precision, f1_score




def prepare(file = 'C:/Users/tianping/Desktop/winequality-red.csv'):

    origin_data = pd.read_csv(file)
    # print(data.columns)
    data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
    data = np.array(data, dtype='float32')
    data[:, 0] = data[:, 0] ** 0.2
    data[:, 1] = data[:, 1] ** 0.4
    data[:, 2] = (data[:, 2] + 1) ** 0.2
    data[:, 3] = data[:, 3] ** 0.5
    data[:, 5] = data[:, 5] ** 0.4
    data[:, 6] = data[:, 6] ** 0.2
    return pd.DataFrame(data, columns=origin_data.columns[0].split(';'))

if __name__ == '__main__':
    d = prepare()
    d_array = np.array(d)

    cls = GaussianOutlierDetection(data=d_array,
                               good_class=[4,5,6,7],
                               bad_class=[1,2,3,8,9,10],
                               epsilon=920,
                               min_feature_nums=3)
    cls.choose_best()

    pprint(cls.ans)

    fpr, tpr, thresholds = metrics.roc_curve(y_true=cls.truth_v2, y_score=1/cls.prob)
    roc_auc = metrics.auc(fpr, tpr)
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
    # plt.figure()
    # plt.hist(d_array[:,10], bins=50)
    # d_array[:,10] = d_array[:,10] **0.001
    # for i in [10]:
    #     plt.figure()
    #     plt.hist(d_array[:, i], bins=50)
    #
    # plt.show()

    # eps = np.arange(1e1, 1e3, 1e1)
    # print(eps)
    # print('---')
    # s = []
    # for k in eps:
    #     cls = GaussianOutlierDetection(data=d_array,
    #                                    good_class=[4,5,6,7],
    #                                    bad_class=[1,2,3,8,9,10],
    #                                    epsilon=k,
    #                                    min_feature_nums=3)
    #     cls.choose_best()
    #     print(list(cls.ans.items())[-1][-1][-1][-1])
    #     s.append(list(cls.ans.items())[-1][-1][-1][-1])
    #     print(k)
    #     print('---------------------------------------')

