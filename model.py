
import tensorflow as tf
from util import FLAGS
import numpy as np
from extra_func import *



class BiRnnAttention(object):
    def __init__(self, sess, q_hidden_dim, a_hidden_dim, num_layers_1, word_vec_initializer,
                 batch_size, vocab_size, embedding_size, learning_rate, margin,
                 query_len_threshold, ans_len_threshold, name, keep_prob=0.5, attention=False):
        self.sess = sess
        self.q_hidden_dim = q_hidden_dim
        self.a_hidden_dim = a_hidden_dim
        self.num_layers_1 = num_layers_1
        self.word_vec_initializer = word_vec_initializer
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.margin = margin
        self.keep_prob = keep_prob
        self.name = name
        self.query_max_len = query_len_threshold
        self.ans_max_len = ans_len_threshold
        self.attention = attention
        self._model()
        self.merged_summary_op = tf.summary.merge([self.loss_op, self.op_merge_summary])

    def _model(self):
        with tf.variable_scope(self.name):
            self.queries_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="queries_len")
            self.pos_ans_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="pos_ans_len")
            self.neg_ans_len = tf.placeholder(dtype=tf.int32, shape=(None,), name="neg_ans_len")
            self.queries = tf.placeholder(dtype=tf.int32, shape=(None, self.query_max_len), name="queries")
            self.pos_answers = tf.placeholder(dtype=tf.int32, shape=(None, self.ans_max_len), name="pos_answers")
            self.neg_answers = tf.placeholder(dtype=tf.int32, shape=(None, self.ans_max_len), name="neg_answers")
            self.queries_mask = tf.placeholder(dtype=tf.bool, shape=(None, self.query_max_len), name="queries_mask")
            self.pos_mask = tf.placeholder(dtype=tf.bool, shape=(None, self.ans_max_len), name="pos_mask")
            self.neg_mask =tf.placeholder(dtype=tf.bool, shape=(None, self.ans_max_len), name="neg_mask")

            with tf.variable_scope("embedding"), tf.device("/cpu:0"):  # as scope:
                initializer = tf.random_uniform_initializer(-1.0, 1.0)
                #with tf.name_scope('embedding_mx'):
                self.embedding_matrix = tf.get_variable(name="embedding_matrix",
                                                        shape=[self.vocab_size, self.embedding_size],
                                                        #initializer=self.word_vec_initializer,  #initializer,
                                                        initializer=initializer,
                                                        dtype=tf.float32)
                op_em_mx_summary = tf.summary.histogram('EmbeddingMatrix', self.embedding_matrix)

                with tf.name_scope('embedded_queries'):
                    self.embedded_queries = tf.nn.embedding_lookup(self.embedding_matrix, self.queries)

                with tf.name_scope('embedded_pos_answers'):
                    self.embedded_pos_answers = tf.nn.embedding_lookup(self.embedding_matrix, self.pos_answers)

                with tf.name_scope('embedded_neg_answers'):
                    self.embedded_neg_answers = tf.nn.embedding_lookup(self.embedding_matrix, self.neg_answers)
                    print("embedding_neg_answers:", tf.shape(self.embedded_neg_answers))

            with tf.name_scope("Training"):
                with tf.name_scope("pre_processing"):
                    self.queries_outputs, self.queries_states= self.bidirectional_rnn(hidden_dim=self.q_hidden_dim,
                                                                  num_layers=self.num_layers_1,
                                                                  input_data=self.embedded_queries,
                                                                  sequence_length=self.queries_len,
                                                                  name='query',
                                                                  keep_prob=self.keep_prob)

                    self.pos_answers_outputs,_ = self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                      num_layers=self.num_layers_1,
                                                                      input_data=self.embedded_pos_answers,
                                                                      sequence_length=self.pos_ans_len,
                                                                      name='answer',
                                                                      keep_prob=self.keep_prob,
                                                                      reuse=False)

                    self.neg_answers_outputs,_ = self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                      num_layers=self.num_layers_1,
                                                                      input_data=self.embedded_neg_answers,
                                                                      sequence_length=self.neg_ans_len,
                                                                      name='answer',
                                                                      keep_prob=self.keep_prob,
                                                                      reuse=True)
                with tf.name_scope("Attention"):
                    self.pos_attention_outputs = self.attention_layer(query_output=self.queries_outputs,
                                                                      ans_output=self.pos_answers_outputs,
                                                                      query_length=self.queries_len,
                                                                      ans_length=self.pos_ans_len,
                                                                      query_mask = self.queries_mask,
                                                                      ans_mask=self.pos_mask,
                                                                      is_training=True,
                                                                      reuse=False)
                    self.neg_attention_outputs = self.attention_layer(query_output=self.queries_outputs,
                                                                      ans_output=self.neg_answers_outputs,
                                                                      query_length=self.queries_len,
                                                                      ans_length=self.neg_ans_len,
                                                                      query_mask=self.queries_mask,
                                                                      ans_mask = self.neg_mask,
                                                                      is_training=True,
                                                                      reuse=True)
                with tf.name_scope("Final_bidirectional_rnn"):
                    _, self.pos_outputs = self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                      num_layers=self.num_layers_1,
                                                                      input_data=self.pos_attention_outputs,
                                                                      sequence_length=self.pos_ans_len,
                                                                      name='attention',
                                                                      keep_prob=self.keep_prob,
                                                                      reuse=False)
                    _, self.neg_outputs = self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                      num_layers=self.num_layers_1,
                                                                      input_data=self.neg_attention_outputs,
                                                                      sequence_length=self.neg_ans_len,
                                                                      name='attention',
                                                                      keep_prob=self.keep_prob,
                                                                      reuse=True)
                with tf.variable_scope('Optimizer'):
                    self.queries_states = self.queries_states[0]  # -1
                    self.pos_outputs = self.pos_outputs[0]
                    self.neg_outputs = self.neg_outputs[0]

                    with tf.name_scope('length_of_tensors'):
                        len1 = tf.sqrt(tf.reduce_sum(tf.multiply(self.queries_states, self.queries_states), 1))
                        len2 = tf.sqrt(
                            tf.reduce_sum(tf.multiply(self.pos_outputs, self.pos_outputs), 1))
                        len3 = tf.sqrt(
                            tf.reduce_sum(tf.multiply(self.neg_outputs, self.neg_outputs), 1))

                    with tf.name_scope('mul_2_tensors'):
                        mul_12 = tf.reduce_sum(tf.multiply(self.queries_states, self.pos_outputs), 1)
                        mul_13 = tf.reduce_sum(tf.multiply(self.queries_states, self.neg_outputs), 1)

                    with tf.name_scope('cos_of_2_tensors'):
                        self.cos_12 = tf.div(mul_12, tf.multiply(len1, len2))
                        self.cos_13 = tf.div(mul_13, tf.multiply(len1, len3))

                zero = tf.constant(0, shape=[self.batch_size], dtype=tf.float32)
                margin = tf.constant(self.margin, shape=[self.batch_size], dtype=tf.float32)

                with tf.name_scope('loss'):
                    self.losses = tf.maximum(zero, tf.subtract(margin, tf.subtract(self.cos_12, self.cos_13)))
                    self.loss = tf.reduce_mean(self.losses)

                self.loss_op = tf.summary.scalar('Loss', self.loss)

                with tf.name_scope("train"):
                    self.train_op = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

            with tf.name_scope("Testing"):
                with tf.name_scope("pre_processing"):
                    self.test_queries_outputs, self.test_queries_states = self.bidirectional_rnn(hidden_dim=self.q_hidden_dim,
                                                                       num_layers=self.num_layers_1,
                                                                       input_data=self.embedded_queries,
                                                                       sequence_length=self.queries_len,
                                                                       name='query',
                                                                       is_training=False,
                                                                       reuse=True)
                    self.test_answers_outputs, _= self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                       num_layers=self.num_layers_1,
                                                                       input_data=self.embedded_pos_answers,
                                                                       sequence_length=self.pos_ans_len,
                                                                       name='answer',
                                                                       is_training=False,
                                                                       reuse=True)
                with tf.name_scope("Attention"):
                    self.test_attention_outputs = self.attention_layer(query_output=self.queries_outputs,
                                                                      ans_output=self.pos_answers_outputs,
                                                                      query_length=self.queries_len,
                                                                      ans_length=self.pos_ans_len,
                                                                      query_mask=self.queries_mask,
                                                                       ans_mask=self.pos_mask,
                                                                      is_training=False,
                                                                      reuse=True)
                with tf.name_scope("Final_bidirectional_rnn"):
                    _, self.test_outputs = self.bidirectional_rnn(hidden_dim=self.a_hidden_dim,
                                                                  num_layers=self.num_layers_1,
                                                                  input_data=self.test_attention_outputs,
                                                                  sequence_length=self.pos_ans_len,
                                                                  name='attention',
                                                                  is_training=False,
                                                                  reuse=True)
                with tf.variable_scope('cos_similarity'):
                    self.test_queries_states = self.test_queries_states[1]
                    self.test_outputs = self.test_outputs[1]

                    len1 = tf.sqrt(tf.reduce_sum(tf.multiply(self.test_queries_states, self.test_queries_states), 1))
                    len2 = tf.sqrt(tf.reduce_sum(tf.multiply(self.test_outputs, self.test_outputs), 1))
                    mul_12 = tf.reduce_sum(tf.multiply(self.test_queries_states, self.test_outputs), 1)

                    self.cos_test = tf.div(mul_12, tf.multiply(len1, len2))

            self.op_merge_summary = tf.summary.merge([op_em_mx_summary])

    def train(self, queries_len, pos_ans_len, neg_ans_len, queries, pos_answers, neg_answers, queries_mask, pos_mask, neg_mask):
        feed_dict = {self.queries_len:queries_len,
                     self.pos_ans_len:pos_ans_len,
                     self.neg_ans_len:neg_ans_len,
                     self.queries:queries,
                     self.pos_answers:pos_answers,
                     self.neg_answers:neg_answers,
                     self.queries_mask:queries_mask,
                     self.pos_mask:pos_mask,
                     self.neg_mask:neg_mask}
        return self.sess.run([self.train_op, self.loss, self.merged_summary_op, self.cos_12, self.cos_13], feed_dict)
    def predict(self, test_queries_len, test_queries, answers, answers_len, queries_mask, ans_mask):
        feed_dict = {self.queries: test_queries,
                    self.queries_len: test_queries_len,
                    self.pos_answers: answers,
                    self.pos_ans_len: answers_len,
                     self.queries_mask: queries_mask,
                     self.pos_mask: ans_mask}

        return self.sess.run([self.cos_test], feed_dict)



    @staticmethod
    def bidirectional_rnn(hidden_dim, num_layers, input_data, sequence_length, name,
                          is_training=True, keep_prob=0.5, reuse=False):
        with tf.variable_scope("Bidirectional-" + name):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            cell = tf.contrib.rnn.LSTMCell(hidden_dim, forget_bias=1.0, state_is_tuple=True)

            if is_training and keep_prob < 1.0:
                cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

            fw_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)
            bw_cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers)

            (outputs, states) = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                cell_bw=bw_cell,
                                                                inputs=input_data,
                                                                sequence_length=sequence_length,
                                                                #time_major=True,
                                                                dtype=tf.float32)
            print("outputs:", outputs)
            outputs = tf.concat(outputs, 2)
            states = tf.concat(states, -1)
            print("states:", states[-1])
            return outputs, states[-1]
    # query_output [?, FLAGS.query_length, 2*hidden_units]
    # ans_output [?, FLAGS.sequence_length, 2*hidden_units]
    def bi_attention(self, query_output, ans_output, query_length, ans_length, query_mask,ans_mask, is_training=True):
        with tf.variable_scope("bi_attention"):
            query_length = np.array(query_length)
            ans_length = np.array(ans_length)
            query_expand_dims = tf.tile(tf.expand_dims(query_output, 1), [1, self.ans_max_len, 1, 1])
            # [query_expand_dims/ans_expand_dims: [?, FLAGS.sequence_length, FLAGS.query_length, 2*hidden_units]
            ans_expand_dims = tf.tile(tf.expand_dims(ans_output, 2), [1, 1, self.query_max_len, 1])
            qmask_expand_dims = tf.tile(tf.expand_dims(query_mask, 1), [1, FLAGS.sequence_length, 1])
            amask_expand_dims = tf.tile(tf.expand_dims(ans_mask, 2), [1, 1, FLAGS.query_length])
            query_ans_mask = qmask_expand_dims & amask_expand_dims

            '''
            if mask:
                query_shape = [query_output.get_shape().as_list()[0], query_output.get_shape().as_list()[1]]
                ans_shape = [ans_output.get_shape().as_list()[0], ans_output.get_shape().as_list()[1]]
                print("query_shape:",query_shape)
                query_mask = np.ones(shape=query_shape, dtype=np.int32)
                ans_mask = np.ones(shape=ans_shape, dtype=np.int32)
                for i in range(query_shape[0]):
                    print("query_length[i]:",query_length)
                    for q in range(query_shape[1]):
                        if q >= query_length[i]:
                            query_mask[i,q] = 0
                    for a in range(ans_shape[1]):
                        if a >=ans_length[i]:
                            ans_mask[i,a] = 0
                qmask_expand_dims = tf.tile(tf.expand_dims(query_mask, 1), [1, FLAGS.sequence, 1])
                amask_expand_dims = tf.tile(tf.expand_dims(ans_mask, 2), [1, 1, FLAGS.query_length])
                query_ans_mask = qmask_expand_dims & amask_expand_dims
            else:
                query_ans_mask = None
            '''

            shared_matrix = get_logits([ans_expand_dims, query_expand_dims], bias=True, mask=query_ans_mask,
                                       wd=FLAGS.wd, input_keep_prob=1.0, is_train=is_training, func=None)
            # shared_matrix [?,FLAGS.sequence_length, FLAGS.query_length]
            c2q = softsel(query_expand_dims, shared_matrix, mask = query_ans_mask) #[?, FLAGS.sequence_length, hidden_units]
            q2c = softsel(ans_output, tf.reduce_max(shared_matrix, 2), mask= None) #[?,hidden_units]
            q2c = tf.tile(tf.expand_dims(q2c, 1), [1, FLAGS.sequence_length, 1])
            return c2q, q2c
    #return [?, FLAGS.sequence_length, 8*hidden_units]
    def attention_layer(self, query_output, ans_output, query_length, ans_length, query_mask, ans_mask, is_training=True, reuse=False):
        with tf.variable_scope("attention_layer"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            if FLAGS.q2c_att or FLAGS.c2q_att:
                c2q, q2c = self.bi_attention(query_output, ans_output, query_length, ans_length, query_mask, ans_mask, is_training=is_training)
            if not FLAGS.c2q_att:
                c2q = tf.tile(tf.expand_dims(tf.reduce_mean(query_output, 1), 1), [1, FLAGS.sequence_length, 1])
            if FLAGS.q2c_att:
                attention_output = tf.concat([ans_output, c2q, ans_output * c2q, ans_output * q2c], axis=2)
            else:
                attention_output = tf.concat([ans_output, c2q, ans_output * c2q],axis=2)
            return attention_output


