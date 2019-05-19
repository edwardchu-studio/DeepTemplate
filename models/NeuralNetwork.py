'''
This file is for neural network definition
'''
import torch

# from torch.nn import ModuleList, Conv2d, ReLU,Linear,MaxPool2d,Softmax2d
# from torch.autograd import Variable
# import torch.nn.functional as F
# import numpy as np
# import torchvision


class NeuralNetwork(torch.nn.Module):
    '''
    NeuralNetwork class
    '''

    def __init__(self, verbose=False):
        '''
        Initialization
        '''
        super(NeuralNetwork, self).__init__()
        self.verbose = verbose
        self._xavier_init_()

    def forward(self, x):
        '''
        Network forward propagation
        '''
        if self.verbose:
            print('..')
        return x

    def _xavier_init_(self):
        '''
        xavier initialization of parameters
        '''
        for sub_module in self:
            if isinstance(sub_module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(sub_module.weight)

if __name__ == '__main__':
    pass