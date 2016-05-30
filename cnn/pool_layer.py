import math
from vol import *
from array_utils import *

class PoolLayer(object):
  def __init__(self,opt):
    # Required
    self.sx = opt.sx # filter size
    self.in_depth = opt.in_depth
    self.in_sx = opt.in_sx
    self.in_sy = opt.in_sy
    #Optional
    self.sy = opt.sy if hasattr(opt,'sy') else opt.sx
    self.stride = opt.stride if hasattr(opt, 'stride') else 2
    self.pad = opt.pad if hasattr(opt,'pad') else 0 # amount of 0 padding to add around borders of input volume
    # computed
    self.out_depth = self.in_depth
    self.out_sx = math.floor((self.in_sx + self.pad * 2 - self.sx) / self.stride + 1)
    self.out_sy = math.floor((self.in_sy + self.pad * 2 - self.sy) / self.stride + 1)
    self.layer_type = 'pool'
    # store switches for x,y coordinates for where the max comes from, for each output neuron
    self.switchx = zeros(self.out_sx*self.out_sy*self.out_depth)
    self.switchy = zeros(self.out_sx*self.out_sy*self.out_depth)

  def forward(self, V, is_training):
    self.in_act = V
    A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)      
    n=0 # a counter for switches
    for d in range(0, int(self.out_depth)):
      x = -self.pad
      y = -self.pad
      for ax in range(0, int(self.out_sx)):
        y = -self.pad
        for ay in range(0, int(self.out_sy)):
          # convolve centered at self particular location
          a = -99999 # hopefully small enough
          winx=-1
          winy=-1
          for fx in range(0, self.sx):
            for fy in range(0, self.sy):
              oy = y+fy
              ox = x+fx # Coordinates wrt original matrix
              if oy>=0 and oy<V.sy and ox>=0 and ox<V.sx :
                v = V.get(ox, oy, d)
                # perform max pooling and store pointers to where
                # the max came from. self will speed up backprop 
                # and can help make nice visualizations in future
                if v > a : 
                  a = v 
                  winx=ox 
                  winy=oy
          self.switchx[n] = winx
          self.switchy[n] = winy
          n += 1
          A.set(ax, ay, d, a)
          y += self.stride
        x += self.stride
    self.out_act = A
    return self.out_act

  def backward(self):
    # pooling layers have no parameters, so simply compute 
    # gradient wrt data here
    V = self.in_act
    V.dw = zeros(len(V.w)) # zero out gradient wrt data
    A = self.out_act # computed in forward pass 
    n = 0
    for d in range(0, self.out_depth):
      x = -self.pad
      y = -self.pad
      for ax in range(0, self.out_sx):
        y = -self.pad
        for ay in range(0, self.out_sy):
          chain_grad = self.out_act.get_grad(ax,ay,d)
          V.add_grad(self.switchx[n], self.switchy[n], d, chain_grad)
          n += 1
          y += self.stride
        x += self.stride
        
  def get_params_and_grads(self):
    return []

