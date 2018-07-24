import tensorflow as tf
from models.base_model import BaseModel

FLAGS = tf.app.flags.FLAGS
def crnn_forward(sent_pos, num_filters1, num_filters2, filter1_size, filter2_size):
    with tf.variable_scope('lstm'):
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_filters1, state_is_tuple=True)
        inputs = tf.unstack(sent_pos, axis = 1)
        outputs, state_fw, state_bw = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=lstm_fw_cell,
                        cell_bw=lstm_bw_cell,
                        inputs = inputs,
                        dtype=tf.float32)
        outputs = tf.stack(outputs, axis = 1)
        h1_rnn = tf.expand_dims(outputs, -1)
        h1_pool = tf.nn.max_pool(h1_rnn,ksize=[1, filter1_size, 1, 1],strides=[1, 1, 1, 1], padding='VALID')

    with tf.variable_scope('cnn'):
        filter_shape = [filter2_size, 2 * num_filters1, 1, num_filters2]
        W_cnn = tf.Variable(tf.truncated_normal(filter_shape, stddev = 0.1), name='W')
        b_cnn = tf.Variable(tf.constant(0.1, shape=[num_filters2]), name="b")
        conv = tf.nn.conv2d(h1_pool, W_cnn, strides = [1,1,1,1], padding = 'VALID', name='conv')
    h1_cnn = tf.nn.relu(tf.nn.bias_add(conv, b_cnn))
    ##Maxpooling
    h2_pool=tf.nn.max_pool(h1_cnn,ksize=[1,seq_len-(filter1_size-1)-(filter2_size-1),1,1],strides=[1, 1, 1, 1],padding="VALID")
    h2_cnn = tf.squeeze(h2_pool, axis=[1,2])
    h_flat = tf.reshape(h2_pool,[-1,num_filters2])
    return h_flat


def linear_layer(name, x, in_size, out_size, is_regularize=False):
    with tf.variable_scope(name):
        loss_l2 = tf.constant(0, dtype=tf.float32)
        w = tf.get_variable('linear_W', [in_size, out_size],
                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('linear_b', [out_size],
                      initializer=tf.constant_initializer(0.1))
        o = tf.nn.xw_plus_b(x, w, b) # batch_size, out_size
        if is_regularize:
            loss_l2 += tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return o, loss_l2

class CRNNModel(BaseModel):
    def __init__(self, word_embed, data, word_dim,
                pos_num, pos_dim, num_relations,
                keep_prob, num_filters1, num_filters2,
                filter1_size, filter2_size,
                lrn_rate, is_train):
        lexical, rid, sentence, pos1, pos2 = data

        w_trainable = True if FLAGS.word_dim == 50 else False
        word_embed = tf.get_variable('word_embed', initializer = word_embed, dtype = tf.float32, trainable = w_trainable)
        pos1_embed = tf.get_variable('pos1_embed', shape = [pos_num, pos_dim])
        pos2_embed = tf.get_variable('pos2_embed', shape = [pos_num, pos_dim])

        self.labels = tf.one_hot(rid, num_relations)
                #embedding lookup
        sentence = tf.nn.embedding_lookup(word_embed, sentence)
        pos1 = tf.nn.embedding_lookup(pos1_embed, pos1)
        pos2 = tf.nn.embedding_lookup(pos2_embed, pos2)

        sent_pos = tf.concat([sentence, pos1, pos2], axis = 2)
        if is_train:
            sent_pos = tf.nn.dropout(sent_pos, keep_prob)

        feature = crnn_forward('crnn', sent_pos, num_filters1, num_filters2)
        feature_size = feature.shape.as_list()[1]
        self.feature = feature
        if is_train:
            feature = tf.nn.dropout(feature, self.dropout_keep_prob)
        logits, loss_l2 = linear_layer('linear_cnn', feature,
                                        feature_size, num_relations,
                                        is_regularize = True)

        prediction = tf.argmax(tf.nn.softmax(logits), 1)
        accuracy = tf.equal(prediction, tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        loss_ce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))
        self.logits = logits
        self.prediction = prediction
        self.accuracy = accuracy
        self.loss = loss_ce + 0.01 * loss_l2
        if not is_train:
            return
        global_step = tf.Variable(0, trainable=False, name='step', dtype = tf.int32)
        optimizer = tf.train.AdamOptimizer(lrn_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step)
        self.global_step = global_step

def build_train_valid_model(word_embed, train_data, test_data):
    with tf.name_scope("Train"):
        with tf.variable_scope('CRNNModel', reuse=None):
            m_train = CRNNModel(word_embed, train_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    FLAGS.keep_prob, FLAGS.num_filters1, FLAGS.num_filters2,
                    FLAGS.filter1_size, FLAGS.filter2_size,
                    FLAGS.lrn_rate, is_train=True)

    with tf.name_scope("Valid"):
        with tf.variable_scope('CRNNModel', reuse=True):
            m_train = CRNNModel(word_embed, test_data, FLAGS.word_dim,
                    FLAGS.pos_num, FLAGS.pos_dim, FLAGS.num_relations,
                    1.0, FLAGS.num_filters1, FLAGS.num_filters2,
                    FLAGS.filter1_size, FLAGS.filter2_size,
                    FLAGS.lrn_rate, is_train=True)
    return m_train, m_valid

