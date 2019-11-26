import tensorflow as tf
import jieba
import numpy as np

from tensorflow.contrib.crf import crf_log_likehood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell

import Parameters as pm
class BiLSTM_CRF():
    def __init__(self, ):
    """Init paramaters"""
        
    def build_model(self):
        # placeholder
        # setting up chars ids and words ids
        self.word_ids = tf.placeholder(tf.int32, shape = [None, None], name = "word_ids")
        self.sequence_lengths = tf.placeholder(tf.int32, shape = [None], name = "sequence_lengths")
        
        self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "labels")
        self.dropout = tf.placeholder(tf.int32, shape = [None], name = "dropout")

        # build look up layer
        with tf.variable_scope("lookup"):
            embedding = tf.Variable(self.embeddings, dtype = tf.float32, traina)
            word_embeddings = tf.nn.embedding_lookup(params = embeddings, 
                                                        ids=self.word_ids,
                                                        name="word_embeddings")
            self.word_embeddings = tf.nn.dropout(word_embeddings, self.dropout)
        # build bilstm
        with tf.variable_scope("bilstm"):
            cell_fw = LSTMCell(self.hidden_dim)
            cell_bw = LSTMCell(self.hidden_dim)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                cell_bw = cell_bw,
                cell_fw = cell_fw,
                inputs = self.word_embeddings,
                sequence_length = self.sequence_lengths,
                dtype = tf.float32
            )
            output = tf.concat([output_fw, output_bw, axis = -1])
            output = tf.nn.dropout(output, self.dropout)
        
        # build project layer
        with tf.variable_scope("project"):
            W = tf.get_variable(
                name = "W",
                shape = [2 * self.hidden_dim, self.num_tags],
                initializer = tf.contrib.layers.xavier_initializer()
            )
