import numpy as np
import tensorflow as tf
from collections import OrderedDict


def map_func(df, unique, length):
    def ingredient_to_idx(unique, ingredients, length):    
        order_dict = OrderedDict({w:unique.index(w) for w in ingredients if w in unique})
        order_list = list(order_dict.values())[:length]
        order_list += [0] * (length - len(order_list))   
        return np.asarray(order_list, dtype=np.int64)
    return df.map(lambda x: ingredient_to_idx(unique, x, length)).values

# string 텐서를 img 텐서로 변환 후 crop
def input_tensor(input_x, label):
    input_x = tf.constant(input_x, dtype=tf.int32)
    input_y = tf.constant(input_x, dtype=tf.float32)
    return input_x, input_y


def batch_iter(data, batch_size, num_epochs, shuffle=False):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


class TextCNN(object):
    """
    <Parameters>
        - sequence_length: 최대 문장 길이
        - num_classes: 클래스 개수
        - vocab_size: 등장 재료 수
        - embedding_size: 각 재료에 해당되는 임베디드 벡터의 차원
        - filter_sizes: convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것인가?) (예: "3, 4, 5")
        - num_filters: 각 filter size 별 filter 수
        - l2_reg_lambda: 각 weights, biases에 대한 l2 regularization 정도
    """

    def __init__(
            self, sequence_length, num_classes, vocab_size,
            embedding_size, filter_sizes, num_filters):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length])
        self.input_y = tf.placeholder(tf.float32, [None, num_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # Embedding layer
        self.We = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(self.We, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        self.h_outputs = []
        self.pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            pad_input = tf.pad(self.embedded_chars_expanded, [[0,0],[filter_size-2,filter_size-2],[0,0],[0,0]], mode='CONSTANT')
            conv = tf.layers.Conv2D(filters=num_filters, kernel_size=[filter_size, embedding_size])(pad_input)
            self.h = tf.nn.relu(conv)
            
            pooled = tf.layers.AveragePooling2D(pool_size=[self.h.shape[1],1],strides=1)(self.h)
            self.h_outputs.append(self.h)
            self.pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(values=self.pooled_outputs, axis=3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        self.scores = tf.layers.dense(self.h_drop, 20)
        
        self.learning_rate = tf.placeholder(tf.float32)
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
        # self.loss = tf.reduce_mean(losses)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=self.input_y, logits=self.scores)
        self.lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, 1000, 0.9, staircase=True)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.extra_update_ops):
            self.adam = tf.train.AdamOptimizer(self.lr_decay).minimize(self.loss,
                                                                        global_step=self.global_step)
            self.sgd = tf.train.GradientDescentOptimizer(self.lr_decay).minimize(self.loss,
                                                                                    global_step=self.global_step)
            self.rms = tf.train.RMSPropOptimizer(self.lr_decay).minimize(self.loss,
                                                                            global_step=self.global_step)
            self.momentum = tf.train.MomentumOptimizer(self.lr_decay, momentum=0.9).minimize(self.loss,
                                                                                                global_step=self.global_step)
    
        self.y_prob = tf.nn.softmax(self.scores)
        self.predictions = tf.argmax(self.scores, 1)

        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        tf.summary.scalar("accuray", self.accuracy)
        tf.summary.scalar("loss", self.loss)
        
        self.merged_summary_op = tf.summary.merge_all()