import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import time
import re
from utils import *
from vol import *
from con_layer import *
from relu_layer import *
from softmax_layer import *
from pool_layer import *
from input_layer import *
from fully_conn_layer import *
from engine import *
from trainer import *
from array_utils import *
from rand_utils import *
import math

def get_idx_from_sent(sent, word_idx_map, max_l, k=300):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l:
        x.append(0)
    return x

def test(net, max_l, k):
    print "loading data...",
    x = cPickle.load(open("test.p","rb"))
    revs, W, word_idx_map= x[0], x[1], x[2]
    print "data loaded!"
    for rev in revs:
        vol = Vol(300,max_l,1)
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k)
        for i in range(0,max_l): # Dependent on sentance length
            for j in range(0,300): # Number of vectors in word2vec 
                vol.w[i*300+j] = W[sent[i]][j]
        net.forward(vol,False)
        print "For tweet:"
        print rev["text"]
        net.print_prediction()
    print "Testing finished"


if __name__=="__main__":
    net = cPickle.load(open('net.p', 'rb'))
    test(net, max_l=32, k=300)