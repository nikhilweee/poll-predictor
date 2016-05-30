import cPickle
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

def train(net, trainer, max_l, k):
    print "loading data...",
    x = cPickle.load(open("train.p","rb"))
    revs, W, word_idx_map= x[0], x[1], x[2]
    print "data loaded!"
    start = time.time()
    i = 0
    for rev in revs:
        i +=1
        vol = Vol(300,max_l,1)
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k)
        for i in range(0,max_l): # Dependent on sentance length
            for j in range(0,300): # Number of vectors in word2vec 
                vol.w[i*300+j] = W[sent[i]][j]
        delta = trainer.train(vol,rev["y"])
        print time.time()-start
        print "Tweet number"
        print i
        start = time.time()
    print "training finished"

def build_cnn(max_l):
    layer_defs = []
    # Input Layer
    d = dictobject({})
    d.type = "input"
    d.out_sx = 300
    d.out_sy = max_l
    d.out_depth = 1
    layer_defs.append(d)
    # Conv Layer 1
    d = dictobject({})
    d.type = "conv"
    d.sx = 5
    d.filters = 3
    d.stride = 2
    d.pad = 0
    d.activation = 'relu'
    layer_defs.append(d)
    # Pool Layer 1
    d = dictobject({})
    d.type = "pool"
    d.sx = 2  
    d.stride = 2
    layer_defs.append(d)
    # Conv Layer 2
    d = dictobject({})   
    d.type = "conv"
    d.sx = 3
    d.filters = 3
    d.stride = 1
    d.pad = 2
    d.activation = 'relu'
    # Pool Layer 2
    layer_defs.append(d)
    d = dictobject({})
    d.type = "pool"
    d.sx = 3  
    d.stride = 3
    layer_defs.append(d)
    # Softmax Layer
    d = dictobject({})
    d.type = 'softmax'
    d.num_classes = 2
    layer_defs.append(d)

    # Initializing neaural network
    net = Net()
    net.make_layers(layer_defs)

    # Define trainer
    d = dictobject({})
    d.batch_size = 20
    d.l2_decay = 0.001

    trainer = Trainer(net,d)
    return net, trainer

if __name__=="__main__":
    net, trainer = build_cnn(max_l=30)
    train(net, trainer, max_l=30, k=300)
    cPickle.dump(net, open('net.p', 'wb'))
