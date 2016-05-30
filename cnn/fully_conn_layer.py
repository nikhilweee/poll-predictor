from vol import *

class FullyConnLayer(object):
    def __init__(self, opt):
        self.out_depth = opt.num_classes
        # optional
        self.l1_decay_mul = opt.l1_decay_mul if hasattr(opt,'l1_decay_mul') else 0.0
        self.l2_decay_mul = opt.l2_decay_mul if hasattr(opt,'l2_decay_mul') else 1.0
        # computed
        self.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth
        self.out_sx = 1
        self.out_sy = 1
        self.layer_type = 'fc'
        # initializations
        bias = opt.bias_pref if hasattr(opt,'bias_pref') else 0.0
        self.filters = []
        for i in range(self.out_depth):
            self.filters.append(Vol(1, 1, self.num_inputs))
        self.biases = Vol(1, 1, self.out_depth, bias)

    def forward(self, V, is_training):
        self.in_act = V
        A = Vol(1, 1, self.out_depth, 0.0)
        Vw = V.w
        for i in range(self.out_depth):
            a = 0.0
            wi = self.filters[i].w
            for d in range(int(self.num_inputs)):
                a += Vw[d] * wi[d] # for efficiency use Vols directly for now
            a += self.biases.w[i]
            A.w[i] = a
        self.out_act = A
        return self.out_act

    def backward(self):
        V = self.in_act
        V.dw = zeros(V.w.length) # zero out the gradient in input Vol
        # compute gradient wrt weights and data
        for i in range(self.out_depth):
            tfi = self.filters[i]
            chain_grad = self.out_act.dw[i]
            for d in range(self.num_inputs):
                V.dw[d] += tfi.w[d]*chain_grad # grad wrt input data
                tfi.dw[d] += V.w[d]*chain_grad # grad wrt params
        self.biases.dw[i] += chain_grad

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
        
