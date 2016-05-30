from array_utils import *

class ReluLayer(object):
    def __init__(self, opt):
        self.out_sx = opt.in_sx
        self.out_sy = opt.in_sy
        self.out_depth = opt.in_depth
        self.layer_type = 'relu'

    def forward(self, V, is_training):
        self.in_act = V
        V2 = V.clone()
        N = len(V.w)
        for x in range(N):
            if V2.w[x] < 0:
                V2.w[x] = 0
        self.out_act = V2
        return self.out_act

    def backward(self):
        V = self.in_act
        V2 = self.out_act
        N = len(V.w)
        V.dw = zeros(N)
        for i in range(N):
            if V2.w[i] <= 0:
                V.dw[i] = 0
            else:
                V.dw[i] = V2.dw[i]
                
    def get_params_and_grads(self):
        return []

