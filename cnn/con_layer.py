import math
from vol import *

class ConvLayer(object):
    def __init__(self, opt):
        self.out_depth = opt.filters
        self.sx = opt.sx
        self.out_depth = opt.filters
        self.sx = opt.sx
        self.in_depth = opt.in_depth
        self.in_sx = opt.in_sx
        self.in_sy = opt.in_sy
        self.sy = opt.sy if hasattr(opt,'sy') else opt.sx
        self.stride = opt.stride if hasattr(opt, 'stride') else 1
        self.pad = opt.pad if hasattr(opt, 'pad') else 0
        self.l1_decay_mul = opt.l1_decay_mul if hasattr(opt,'l1_decay_mul') else 0.0
        self.l2_decay_mul = opt.l2_decay_mul if hasattr(opt,'l2_decay_mul') else 1.0
        self.out_sx = math.floor((self.in_sx + self.pad * 2 - self.sx) / self.stride + 1)
        self.out_sy = math.floor((self.in_sy + self.pad * 2 - self.sy) / self.stride + 1)
        self.layer_type = 'conv'
        bias = opt.bias_pref if hasattr(opt, 'bias_pref') else 0.0
        self.filters = []
        for i in range(self.out_depth):
          self.filters.append(Vol(self.sx, self.sy, self.in_depth))
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(self.out_sx, self.out_sy, self.out_depth, 0.0)
        for d in range(self.out_depth):
            f = self.filters[d]
            x = -self.pad
            y = -self.pad
            for ax in range(int(self.out_sx)):
                y = -self.pad
                for ay in range(int(self.out_sy)):
                    a = 0.0
                    for fx in range(f.sx):
                        for fy in range(f.sy):
                            for fd in range(f.depth):
                                oy = y+fy
                                ox = x+fx
                                if oy>=0 and oy<V.sy and ox>=0 and ox<V.sx:
                                    a += f.w[int(((f.sx * fy)+fx)*f.depth+fd)] * V.w[int(((V.sx * oy)+ox)*V.depth+fd)]
                    a += self.biases.w[d]
                    A.set(ax, ay, d, a)
                    y += self.stride
                x += self.stride
        self.out_act = A
        return self.out_act

    def backward(self):
        V = self.in_act
        V.dw = zeros(len(V.w))
        for d in range(self.out_depth):
            f = self.filters[d]
            x = -self.pad
            y = -self.pad
            for ax in range(self.out_sx):
                y = -self.pad
                for ay in range(self.out_sy):
                    chain_grad = self.out_act.get_grad(ax,ay,d)
                    for fx in range(f.sx):
                        for fy in range(f.sy):
                            for fd in range(f.depth):
                                oy = y+fy
                                ox = x+fx
                                if oy>=0 and oy<V.sy and ox>=0 and ox<V.sx:
                                    ix1 = ((V.sx * oy)+ox)*V.depth+fd
                                    ix2 = ((f.sx * fy)+fx)*f.depth+fd
                                    f.dw[ix2] += V.w[ix1]*chain_grad
                                    V.dw[ix1] += f.w[ix2]*chain_grad
                    y += self.stride
                    self.biases.dw[d] += chain_grad
                ax += self.stride

    def get_params_and_grads(self):
        response = []
        for i in range(self.out_depth):
            d = dictobject({})
            d.params = self.filters[i].w
            d.grads = self.filters[i].dw
            d.l2_decay_mul = self.l2_decay_mul
            d.l1_decay_mul = self.l1_decay_mul
            response.append(d)
        d = dictobject({})
        d.params = self.biases.w
        d.grads = self.biases.dw
        d.l1_decay_mul = 0.0
        d.l2_decay_mul = 0.0
        response.append(d)
        return response

