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

class Net(object):
    def __init__(self):
        self.layers = []

    # desugar syntactic for adding activations and dropouts
    def desugar(self, defs):
        new_defs = []
        for i in range(0, len(defs)):
            defination = defs[i]          
            if defination.type =='softmax':
                # add an fc layer here, there is no reason the user should
                # have to worry about this and we almost always want to
                d = dictobject({})
                d.type='fc'
                d.num_classes = defination.num_classes
                new_defs.append(d)
            if (defination.type =='fc' or defination.type =='conv') and not hasattr(defination,"bias_pref"):
                defination.bias_pref = 0.0
                defination.bias_pref = 0.1 # relus like a bit of positive bias to get gradients early
                    # otherwise it's technically possible that a relu unit will never turn on (by chance)
                    # and will never get any gradient and never contribute any computation. Dead relu.

            new_defs.append(defination)
            if hasattr(defination, 'activation') :
                if defination.activation =='relu':
                    d = dictobject({})
                    d.type = 'relu'
                    new_defs.append(d)
                else:
                    print('ERROR unsupported activation ' + defination.activation)
        return new_defs
    #takes a list of layer definitions and creates the network layer objects
    def make_layers(self, defs):
        # few checks for now
        if len(defs)<2 :
            print('ERROR! For now at least have input and softmax layers.')
        if defs[0].type != 'input':
            print('ERROR! For now first layer should be input.')
        defs = self.desugar(defs)
        # create the layers
        self.layers = []
        for i in range(0,len(defs)):
            defination = defs[i]
            if i>0:
                prev = self.layers[i-1]
                defination.in_sx = prev.out_sx
                defination.in_sy = prev.out_sy
                defination.in_depth = prev.out_depth
            if defination.type == 'fc':
                self.layers.append(FullyConnLayer(defination))    
            elif defination.type == 'softmax':
                self.layers.append(SoftmaxLayer(defination))
            elif defination.type == 'input':
                self.layers.append(InputLayer(defination))
            elif defination.type == 'conv':
                self.layers.append(ConvLayer(defination))
            elif defination.type == 'pool':
                self.layers.append(PoolLayer(defination))              
            elif defination.type == 'relu':
                self.layers.append(ReluLayer(defination))
            else :
                print('ERROR: UNRECOGNIZED LAYER TYPE!')

    # forward prop the network. A trainer will pass in is_training = true
    def forward(self, V, is_training):
        if not is_training:
            is_training = False
        act = self.layers[0].forward(V, is_training)
        for i in range(1,len(self.layers)):
            act = self.layers[i].forward(act, is_training)
        return act

    # backprop: compute gradients wrt all parameters
    def backward(self, y):
        N = len(self.layers)
        loss = self.layers[N-1].backward(y) # last layer assumed softmax
        for i in range(N-2,-1): # first layer assumed input
            self.layers[i].backward()
        return loss

    def get_params_and_grads(self):
        # accumulate parameters and gradients for the entire network
        response = []
        for i in range(0,len(self.layers)):
            layer_reponse = self.layers[i].get_params_and_grads()
            for j in range(0,len(layer_reponse)):
                response.append(layer_reponse[j])
        return response
    
    def get_prediction(self):
        S = self.layers[len(self.layers)-1] # softmax layer
        p = S.out_act.w
        maxv = p[0]
        maxi = 0
        for i in range(1,len(p)):
            if p[i] > maxv :
                maxv = p[i]
                maxi = i
        return maxi

    def print_prediction(self):
        S = self.layers[len(self.layers)-1] # softmax layer
        p = S.out_act.w
        print "Probablity of negative is:"
        print p[0]
        print "Probablity of positive is:"  
        print p[1]      
