import numpy as np
import tensorflow as tf

# from tensorflow.contrib import rnn

from model import Model
from utils.language_utils import letter_to_vec, word_to_indices


class ClientModel(Model):
    def __init__(self, lr, seq_len, num_classes, n_hidden, n_lstm_layers=2, max_batch_size=None, seed=None):
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.n_hidden = n_hidden
        self.n_lstm_layers = n_lstm_layers
        super(ClientModel, self).__init__(lr, seed, max_batch_size)

    def create_model(self):
        # features = tf.placeholder(tf.int32, [None, self.seq_len])
        features = tf.compat.v1.placeholder(tf.int32, [None, self.seq_len])
        # embedding = tf.get_variable("embedding", [self.num_classes, 8])
        embedding = tf.compat.v1.get_variable("embedding", [self.num_classes, 8])
        x = tf.nn.embedding_lookup(embedding, features)
        # labels = tf.placeholder(tf.int32, [None, self.num_classes])
        labels = tf.compat.v1.placeholder(tf.int32, [None, self.num_classes])

        stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(
            [tf.compat.v1.nn.rnn_cell.BasicLSTMCell(self.n_hidden) for _ in range(self.n_lstm_layers)])
        outputs, _ = tf.compat.v1.nn.dynamic_rnn(stacked_lstm, x, dtype=tf.float32)
        pred = tf.compat.v1.layers.dense(inputs=outputs[:, -1, :], units=self.num_classes)

        loss = tf.reduce_mean(
            tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=labels))
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.compat.v1.train.get_global_step())

        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
        eval_metric_ops = tf.math.count_nonzero(correct_pred)

        return features, labels, loss, train_op, eval_metric_ops

    def process_x(self, raw_x_batch):
        x_batch = [word_to_indices(word) for word in raw_x_batch]
        x_batch = np.array(x_batch)
        return x_batch

    def process_y(self, raw_y_batch):
        y_batch = [letter_to_vec(c) for c in raw_y_batch]
        return y_batch
