import tensorflow as tf
import numpy as np
import tqdm
from sklearn import preprocessing
import pandas as pd
from tqdm import tqdm


class network:

    def __init__(self, data=None, train_num=1300, val_num=200, batch_size=32, epoch=50, lr=1e-4, prop=0.5):

        self.batch_size = batch_size
        self.epoch = epoch
        self.data = data
        self.num_batch = self.data.shape[0] // self.batch_size
        self.train_num = train_num
        self.val_num = val_num
        self.lr = lr
        self.prop = prop

    def build(self):

        self.keep_prop = tf.placeholder(dtype=tf.float32, name='keep_prop')
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 11], name='x')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 8], name='y')
        self.is_train = tf.placeholder(dtype=tf.bool, name='is_train')
        self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')

        with tf.variable_scope('inputs_hidden1'):

            hidden1 = tf.layers.dense(self.x, 100, activation=tf.nn.relu, name='dense_1')
            hidden1 = tf.nn.dropout(hidden1, keep_prob=self.keep_prop)

        hidden2 = self.dense(hidden1, 64, 'hidden2')

        with tf.variable_scope('hidden4_outputs'):

            self.logits = tf.layers.dense(hidden2, 8)
            self.outputs = tf.nn.softmax(self.logits)

        with tf.name_scope('accuracy'):

            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.outputs, axis=1),
                                                    tf.argmax(self.y, axis=1)), dtype=tf.float32))

        with tf.name_scope('recall'):
            self.recall = tf.constant(0, dtype=tf.float32)

            def f1():
                return tf.reduce_sum(tf.cast(tf.equal(tf.cast(TP_and_TN, dtype=tf.float32)+tf.cast(class_i,dtype=tf.float32),
                                     tf.constant(2, dtype=tf.float32)),dtype=tf.float32))/ tf.reduce_sum(tf.cast(class_i, dtype=tf.float32))
            def f2():
                return tf.constant(0, dtype=tf.float32)
            for i in range(1, 9):

                TP_and_TN = tf.equal(tf.argmax(self.outputs, axis=1),
                                                tf.argmax(self.y, axis=1))
                class_i = tf.equal(tf.argmax(self.y, axis=1), tf.constant(i, dtype=tf.int64))
                self.recall += tf.cond(tf.less(tf.constant(0, dtype=tf.float32),
                                               tf.reduce_sum(tf.cast(class_i, dtype=tf.float32))),
                    f1, f2)

            self.recall /= 4

        with tf.name_scope('optimize'):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                                      logits=self.logits))
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('accuracy', self.accuracy)
        tf.summary.scalar('recall', self.recall)
        self.merged = tf.summary.merge_all()

    def dense(self, inputs, units, name=None):

        with tf.variable_scope(name):

            z = tf.layers.dense(inputs=inputs, units=units, activation=None)
            z_norm = tf.layers.batch_normalization(z, training=self.is_train)
            a_relu = tf.nn.relu(z_norm)
            a_dropout = tf.nn.dropout(a_relu, keep_prob=self.keep_prop)

        return a_dropout

    def prepare_data(self):

        train_x = self.data[range(self.train_num), :-1]
        train_y = self.data[range(self.train_num), -1]
        val_x = self.data[range(self.train_num, self.train_num+self.val_num), :-1]
        val_y = self.data[range(self.train_num, self.train_num+self.val_num), -1]

        train_x_norm = preprocessing.minmax_scale(train_x)
        val_x_norm = preprocessing.minmax_scale(val_x)
        train_y_one_hot = preprocessing.label_binarize(train_y, classes=np.arange(1, 9))
        val_y_one_hot = preprocessing.label_binarize(val_y, classes=np.arange(1, 9))

        return train_x_norm, train_y_one_hot, val_x_norm, val_y_one_hot

    def reset(self):

        self.current_index = 0


    def next_batch(self):

        assert self.current_index < self.train_x_norm.shape[0]
        batch_x = self.train_x_norm[self.current_index:(self.current_index + self.batch_size), :]
        batch_y = self.train_y[self.current_index:(self.current_index + self.batch_size)]

        return batch_x, batch_y

    def train(self, sess):

        (self.train_x_norm, self.train_y, self.val_x_norm, self.val_y) = self.prepare_data()
        index = np.arange(0, self.train_x_norm.shape[0], 1)

        sess.run(tf.global_variables_initializer())
        writer_train = tf.summary.FileWriter('G:/python_file/wine/train/', sess.graph)
        writer_val = tf.summary.FileWriter('G:/python_file/wine/val/', sess.graph)
        num = 0

        for epo in tqdm(range(self.epoch), desc='epoch'):

            np.random.shuffle(index)
            self.train_x_norm = self.train_x_norm[index, :]
            self.train_y = self.train_y[index]

            self.reset()

            for _ in range(self.num_batch):

                num += 1
                (batch_x, batch_y) = self.next_batch()

                feed_dict = {self.x: batch_x, self.y: batch_y, self.keep_prop: self.prop,
                             self.is_train: True, self.learning_rate: self.lr}
                _, train_result = sess.run([self.train_op, self.merged], feed_dict=feed_dict)
                writer_train.add_summary(train_result, num)

                feed_dict_val = {self.x: self.val_x_norm, self.y: self.val_y, self.keep_prop: 1.0,
                                 self.is_train: False}
                val_result = sess.run(self.merged, feed_dict=feed_dict_val)
                writer_val.add_summary(val_result, num)

            if epo % 1000 == 0:
                self.lr *= 0.6

def prepare(file = 'C:/Users/tianping/Desktop/winequality-red.csv'):

    origin_data = pd.read_csv(file)
    # print(data.columns)
    data = [origin_data.ix[i, 0].split(';') for i in range(origin_data.shape[0])]
    data = np.array(data, dtype='float32')

    for index, class_ in enumerate(data[:, -1]):

        if class_ in [5, 6, 7]:
            data[index, -1] = 5
        elif class_ in [8, 9, 10]:
            data[index, -1] -= 2

    return pd.DataFrame(data, columns=origin_data.columns[0].split(';'))



if __name__ == '__main__':
    data = prepare()
    data_array = np.array(data)
    sess = tf.Session()
    net = network(data=data_array)
    net.build()
    net.train(sess)


