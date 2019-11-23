import tensorflow as tf
import jieba
import numpy as np

from tensorflow.contrib.crf import crf_log_likehood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell

class BiLSTM_CRF():
    def __init__(self):
    """Init paramaters"""
        
    def build_model(self):
        # placeholder
        self.word_ids = tf.placeholder(tf.int32, shape = [None, None], name = "word_ids")
        self.labels = tf.placeholder(tf.int32, shape = [None, None], name = "labels")
        self.sequence_lengths = tf.placeholder(tf.int32, shape = [None], name = "sequence_lengths")
        self.dropout = tf.placeholder(tf.int32, shape = [None], name = "dropout")

        # build look up layer
        with tf.variable_scope("lookup"):
            