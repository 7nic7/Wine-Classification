import tensorflow as tf
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import roc_curve,auc
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


class network:

    def __init__(self, batch_size=32, epoch=50, lr=1e-4,
                 max_lr=0.02, enlarge_lr=1.005, reduce_lr=0.98):

        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.max_lr = max_lr
        self.enlarge_lr = enlarge_lr
        self.reduce_lr = reduce_lr
        self.e = 0

    def build(self, a2, a3, a4):

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x_layer1')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='y_layer5')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        with tf.variable_scope('inputs_1'):

            layer2 = tf.layers.dense(self.x, a2, activation=tf.nn.tanh, name='layer2')

        with tf.variable_scope('encoder'):

            layer3 = tf.layers.dense(layer2, a3, name='layer3')  # can't change
            encoder = step_wise(layer3, n=a3)         # the most important

        with tf.variable_scope('outputs_5'):

            layer4 = tf.layers.dense(encoder, a4, activation=tf.nn.tanh, name='layer4')
            self.outputs = tf.layers.dense(layer4, 11, activation=tf.nn.sigmoid, name='layer5')

        with tf.name_scope('optimize'):

            self.loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y,
                                                                    predictions=self.outputs))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

    def reset(self):

        index = np.arange(0, self.train_X.shape[0], 1)
        np.random.shuffle(index)
        self.train_X = self.train_X[index, :]
        # self.train_y = self.train_y[index]
        self.current_index = 0

    def next_batch(self):

        assert self.current_index < self.train_X.shape[0]
        batch_x = self.train_X[self.current_index:(self.current_index + self.batch_size), :]
        batch_y = batch_x

        return batch_x, batch_y

    def train(self, X, val_X, sess=None):

        self.train_X = X
        num_batch = self.train_X.shape[0] // self.batch_size
        sess.run(tf.global_variables_initializer())
        self.writer_train = tf.summary.FileWriter('G:/python_file/wine/train/', sess.graph)
        self.writer_val = tf.summary.FileWriter('G:/python_file/wine/val/', sess.graph)
        num = 0
        for _ in tqdm(range(self.epoch), desc='epoch'):
        # for _ in range(self.epoch):
            self.reset()
            for _ in range(num_batch):

                num += 1
                (batch_x, batch_y) = self.next_batch()

                feed_dict = {self.x: batch_x, self.y: batch_y,
                             self.learning_rate: self.lr}
                _, e, train_result = sess.run([self.train_op, self.loss, self.merged],
                                              feed_dict=feed_dict)
                val_result = sess.run(self.merged, feed_dict={self.x:val_X, self.y:val_X})
                self.writer_train.add_summary(train_result, num)
                self.writer_val.add_summary(val_result, num)
                if e > 1.01 * self.e:
                    self.lr *= self.reduce_lr
                elif e < self.e and self.lr < self.max_lr:
                    self.lr *= self.enlarge_lr
                else:
                    self.lr = self.lr
                self.e = e


def step_wise(theta, a=100, n=4):   # theta : tensor with shape of 32,
    out = tf.zeros_like(theta)
    for i in range(1, n):
        out = tf.add(tf.nn.tanh(a*(theta-tf.constant(i/n))), out)
    out = 1/2 + 1/4 * out

    return out


def split_data(data, train_ratio, good_class=[4,5,6,7], bad_class=[1,2,3,8,9,10], shuffle=False):

    good_example = np.concatenate([data[index,:] for index,class_ in enumerate(data[:, -1])\
                                   if class_ in good_class]).reshape([-1, 12])
    good_example_nums = good_example.shape[0]
    bad_example = np.concatenate([data[index,:] for index,class_ in enumerate(data[:,-1])\
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


def prepare(file='C:/Users/tianping/Desktop/winequality-red.csv'):

    origin_data = pd.read_csv(file)
    # print(data.columns)
    data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
    data = np.array(data, dtype='float32')
    data[:, :-1] = preprocessing.minmax_scale(data[:, :-1], axis=0)

    return pd.DataFrame(data, columns=origin_data.columns[0].split(';'))


if __name__ == '__main__':
    data = prepare()
    data_array = np.array(data)
    train_X, train_y, val_X, val_y, test_X, test_y = split_data(data= data_array,
                                                                    train_ratio=0.6)
    val_num = val_X.shape[0]
    val_labels = np.zeros(shape=[val_num, 1])

    for i, class_ in enumerate(val_y):
        if class_ in [1, 2, 3, 8, 9, 10]:
            val_labels[i] = 1


    tf.reset_default_graph()
    sess = tf.Session()

    net = network(epoch=200, lr=1e-4)
    net.build(a2=9, a3=4, a4=9)


    net.train(X=train_X,val_X=val_X,sess=sess)
    val_output = sess.run(net.outputs,
                      feed_dict={net.x: val_X})

    val_dist = np.zeros(shape=[val_num, 1])
    for i, x in enumerate(val_X):
        val_dist[i] = np.linalg.norm(x-val_output[i])   # euclidean distance

    fpr, tpr, thresholds = roc_curve(val_labels, val_dist)
    roc_auc = auc(fpr, tpr)
    print('val auc:%s'%(roc_auc))

    test_num = test_X.shape[0]
    test_dist = np.zeros(shape=[test_num, 1])

    test_output = sess.run(net.outputs, feed_dict={net.x:test_X})
    for i, x in enumerate(test_X):
        test_dist[i] = np.linalg.norm(x-test_output[i])   # euclidean distance

        # test_dists.append(test_dist)

    # dists = np.mean(np.concatenate(test_dists, axis=1), axis=1)
    test_labels = np.zeros(shape=[test_num, 1])
    for i, class_ in enumerate(test_y):
        if class_ in [1, 2, 3, 8, 9, 10]:
            test_labels[i] = 1

    fpr, tpr, thresholds = roc_curve(test_labels, test_dist)
    test_roc_auc = auc(fpr, tpr)
    print('test auc: %s'%test_roc_auc)

