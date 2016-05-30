class InputLayer(object):
    def __init__(self, opt):
        if opt.out_sx:
            self.out_sx = opt.out_sx
        else:
            self.out_sx = opt.in_sx
        if opt.out_sy:
            self.out_sy = opt.out_sy
        else:
            self.out_sy = opt.in_sy
        if opt.out_depth:
            self.out_depth = opt.out_depth
        else:
            self.out_depth = out.in_depth
        self.layer_type = 'input'

    def forward(self, V, is_training):
        self.in_act = V
        self.out_act = V
        return self.out_act

    def backward(self):
        pass

    def get_params_and_grads(self):
        return []