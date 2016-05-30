from array_utils import *
from rand_utils import *
import math

class Vol(object):
  def __init__(self, sx, sy, depth, c=None):
    self.sx = sx
    self.sy = sy
    self.depth = depth
    n = int(sx*sy*depth)
    self.w = zeros(n)
    self.dw = zeros(n)

    if not c:
      #  weight normalization
      scale = math.sqrt(1.0/(sx*sy*depth))
      for i in range(n):
        self.w[i] = randn(0.0, scale)
    else:
      for i in range(n):
        self.w[i] = c

  def get_coordinate(self, x,y,d):
    return int(((self.sx * y)+x)*self.depth+d)

  def get(self, x, y, d):
    return self.w[self.get_coordinate(x,y,d)]

  def set(self, x, y, d, v):
    self.w[self.get_coordinate(x,y,d)] = v

  def add(self, x, y, d, v):
    self.w[self.get_coordinate(x,y,d)] += v

  def get_grad(self, x, y, d):
    return self.dw[self.get_coordinate(x,y,d)]

  def set_grad(self, x, y, d, v):
    self.dw[self.get_coordinate(x,y,d)] = v

  def add_grad(self, x, y, d, v):
    self.dw[self.get_coordinate(x,y,d)] += v

  def clone_and_zero(self):
    return Vol(self.sx, self.sy, self.depth, 0.0)

  def clone(self, ):
    V = Vol(self.sx, self.sy, self.depth, 0.0)
    V.w = self.w[:]
    return V
  def add_from(self, V):
    for k in range(len(self.w)):
      self.w[k] += V.w[k]
  def add_from_scaled(self, V, a):
    for k in range(len(self.w)):
      self.w[k] += a*V.w[k]
  def set_const(self, a):
    for k in range(len(self.w)):
      self.w[k] = a
