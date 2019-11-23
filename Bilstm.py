import tensorflow as tf
import jieba
import numpy as np

from tensorflow.contrib.crf import crf_log_likehood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.rnn import LSTMCell

class BiLSTM_CRF():
    def __init__(self):
    """Init paramaters"""
        self.batch_size = 0
        self.epoch = 0
        self.hidden_dim = 0
        self.embeddings = 0
    def 