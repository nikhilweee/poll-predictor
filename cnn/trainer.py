import datetime
from utils import *
from array_utils import *
import math

class Trainer(object):
  def __init__(self,net,options):
    self.learning_rate = options.learning_rate if hasattr(options,'learning_rate') else 0.01
    self.l1_decay = options.l1_decay if hasattr(options,'l1_decay') else 0.0
    self.l2_decay = options.l2_decay if hasattr(options,'l2_decay') else 0.0
    self.batch_size = options.batch_size if hasattr(options,'batch_size') else 1
    self.momentum = options.momentum if hasattr(options,'momentum') else 0.9
    self.eps = 0.000001
    self.k = 0 # iteration counter
    self.gsum = [] # last iteration gradients (used for momentum calculations)
    self.net = net

  def train(self, x, y):
    start = datetime.datetime.now()
    self.net.forward(x, True) # also set the flag that lets the net know we're just training
    end = datetime.datetime.now()
    fwd_time = end - start
    start = datetime.datetime.now()
    cost_loss = self.net.backward(y)
    l2_decay_loss = 0.0
    l1_decay_loss = 0.0
    end = datetime.datetime.now()
    bwd_time = end - start      
    self.k += 1
    if self.k % self.batch_size == 0 :
      pglist = self.net.get_params_and_grads()
      # initialize lists for accumulators. Will only be done once on first iteration
      if len(self.gsum) == 0:
        # adagrad needs gsum
        for i in range(0,len(pglist)):
          self.gsum.append(zeros(len(pglist[i].params)))
      # perform an update for all sets of weights
      for i in range(0,len(pglist)):
        pg = pglist[i] # param, gradient, other options in future (custom learning rate etc)
        p = pg.params
        g = pg.grads
        # learning rate for some parameters.
        l2_decay_mul = pg.l2_decay_mul if hasattr(pg,'l2_decay_mul') else 1.0
        l1_decay_mul = pg.l1_decay_mul if hasattr(pg,'l1_decay_mul') else 1.0
        l2_decay = self.l2_decay * l2_decay_mul
        l1_decay = self.l1_decay * l1_decay_mul
        plen = len(p)
        for j in range(0,plen):
          l2_decay_loss += l2_decay*p[j]*p[j]/2 # accumulate weight decay loss
          l1_decay_loss += l1_decay*math.fabs(p[j])
          l1grad = l1_decay * (1 if p[j] > 0 else -1)
          l2grad = l2_decay * (p[j])
          gij = (l2grad + l1grad + g[j]) / self.batch_size # raw batch gradient
          gsumi = self.gsum[i]
          # adagrad update
          gsumi[j] = gsumi[j] + gij * gij
          dx = - self.learning_rate / math.sqrt(gsumi[j] + self.eps) * gij
          p[j] += dx
          g[j] = 0.0 # zero out gradient so that we can begin accumulating anew

    # appending softmax_loss for backwards compatibility, but from now on we will always use cost_loss
    # in future, TODO: have to completely redo the way loss is done around the network as currently 
    # loss is a bit of a hack. Ideally, user should specify arbitrary number of loss functions on any layer
    # and it should all be computed correctly and automatically. 
    d = dictobject({})
    d.fwd_time = fwd_time
    d.bwd_time = bwd_time
    d.l2_decay_loss = l2_decay_loss
    d.l1_decay_loss = l1_decay_loss
    d.cost_loss = cost_loss
    d.loss = cost_loss + l1_decay_loss + l2_decay_loss
    
    return d