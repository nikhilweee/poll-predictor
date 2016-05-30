from vol import *

class SoftmaxLayer(object):
  def __init__(self, opt):
    # computed
    self.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
    self.out_depth = self.num_inputs
    self.out_sx = 1
    self.out_sy = 1
    self.layer_type = 'softmax'

  def forward(self, V, is_training):
    self.in_act = V
    A = Vol(1, 1, self.out_depth, 0.0)
    # compute max activation
    vol_list = V.w
    amax = V.w[0]
    for i in range(1, self.out_depth):
      if vol_list[i] > amax: 
        amax = vol_list[i]
    # compute exponentials (carefully to not blow up)
    es = zeros(self.out_depth)
    esum = 0.0
    for i in range(0, self.out_depth):
      e = math.exp(vol_list[i] - amax)
      esum += e
      es[i] = e
    # normalize and output to sum to one
    for i in range(0, self.out_depth):
      es[i] /= esum
      A.w[i] = es[i]
    self.es = es  # save these for backprop
    self.out_act = A
    return self.out_act

  def backward(self,y):
    # compute and accumulate gradient wrt weights and bias of this layer
    x = self.in_act
    x.dw = zeros(len(x.w)) # zero out the gradient of input Vol
    for i in range(0, self.out_depth):
      indicator = 1.0 if i == y else 0.0
      mul = -(indicator - self.es[i])
      x.dw[i] = mul
    # loss is the class negative log likelihood
    return -math.log(self.es[y])

  def get_params_and_grads(self):
    return []

