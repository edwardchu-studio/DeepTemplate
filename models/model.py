'''
This is the model file where the logic of training, testing, predicting, etc. should be implemented.
'''
#===================================================
# system wise package
import shutil
import os
# import sys
# import pickle
# import pprint
# import collections
# import itertools
import datetime
# import yaml
import json
#====================================================
# deep-learning package
import torch
from torch.autograd import Variable
import torchvision
from tensorboardX import SummaryWriter
# import numpy as np
# import matplotlib.pyplot as plt
#=====================================================
# model wise customized module

#====================================================


class Model:
    '''
    Class where learning happends
    '''

    def __init__(self, args):
        self.gpu = torch.cuda.is_available()
        self.args = args
        if self.gpu:
            self.command_log('GPU detected, using cuda')
            self.network = torchvision.models.vgg16().cuda()
        else:
            self.command_log('GPU not detected, using cpu')
            self.network = torchvision.models.vgg16()
        self.optimizer = {
            'adam':
            torch.optim.Adam(self.network.parameters(), lr=self.args['alpha']),
            'adadelta':
            torch.optim.Adadelta(self.network.parameters(), lr=self.args['alpha']),
            'rms':
            torch.optim.RMSprop(self.network.parameters()),
            'adagrad':
            torch.optim.Adagrad(self.network.parameters(), lr=self.args['alpha'])
        }.get(self.args['optimizer_name'],
              torch.optim.SGD(self.network.parameters(), lr=self.args['alpha']))
        # self.dataloader = DataLoader()

        self.data_set = {}
        self.checkpoint_path = '../checkpoints'
        self.logger = SummaryWriter('../logs/log_' + self.args['tag'] + '_' +
                                    datetime.datetime.now().strftime('%D-%T').
                                    replace('/', '_') + json.dumps(args))

    def data_preprocess(self):
        '''
        Data preprocess.
        Jobs done here should be different from what may be done inside customized DataLoader,
        in that some preprocess jobs may need information of input or network,
        which can't be done using just data file.
        '''
        pass

    def calculate_loss(self, batch_y_, batch_y):
        '''
        This is where loss is calculated
        '''
        loss = batch_y-batch_y_
        acc = batch_y-batch_y_
        return loss, acc

    def run(self):
        '''
        Network gets trained, developed, tested here.
        '''
        global_best_loss = float('inf')

        if self.args['resume']:
            self.load_check_point()

        for epoch_id in range(self.args['epoch_num']):

            for phase in ['TRAIN', 'TEST']:
                self.command_log('{}/{} epoch, {}.'.format(
                    epoch_id + 1, self.args['epoch_num'], phase))
                if phase == 'TRAIN':
                    train = True
                    self.network.train(True)
                else:
                    train = False
                    self.network.train(False)

                cur_batch_len = len(self.data_set[phase]['X'])

                for batch_id, data in enumerate(
                        zip(self.data_set[phase]['X'], self.data_set[phase]['Y'])):
                    batch_x, batch_y = data
                    batch_x = Variable(torch.Tensor(batch_x).float())
                    if self.gpu:
                        batch_x = batch_x.cuda()
                    if train:
                        self.optimizer.zero_grad()
                    batch_y_ = self.network(batch_x)

                    loss, acc = self.calculate_loss(batch_y_, batch_y)

                    self.logger.add_scalar(phase + 'loss/', loss,
                                           epoch_id * cur_batch_len + batch_id)

                    self.logger.add_scalar(
                        'lr', self.optimizer.param_groups[0]['lr'],
                        epoch_id * len(self.data_set[phase]['X']) + batch_id)

                    self.command_log(
                        f'epoch:{epoch_id},batch:{batch_id},{phase}. Loss:{loss}, Acc:{acc}'
                    )

                    if loss < global_best_loss:
                        self.command_log('Best Updated')
                        global_best_loss = loss
                        self.save_check_point(True)

                    if cur_batch_len % (batch_id + 1) == 0:
                        self.save_check_point()

                    if train:
                        loss.backward()
                        for tag, value in self.network.named_parameters():
                            tag = tag.replace('.', '/')
                            self.logger.add_histogram(
                                tag,
                                value.data.cpu().numpy(),
                                epoch_id * cur_batch_len + batch_id)
                            if hasattr(value.grad, 'data'):
                                self.logger.add_histogram(
                                    tag + '/grad',
                                    value.grad.data.cpu().numpy(),
                                    epoch_id * cur_batch_len + batch_id)
                        self.optimizer.step()

    def test(self):
        '''
        Write Test Code Here.
        Test on test_dataset should be done in function `run()`,
        whereas this function is for scratching and debugging.
        '''
        pass

    def save_check_point(self, best=False, filename='checkpoint.pth.tar'):
        '''
        Saving checkpoints.
        If current state to save is the best,
        the function will copy the saved file and tag it best.
        '''
        torch.save(self.network.state_dict(),
                   self.checkpoint_path + '/' + filename)
        if best:
            shutil.copyfile(self.checkpoint_path + '/' + filename,
                            self.checkpoint_path + '/' + 'best.pth.tar')

    def load_check_point(self, best=False, filename='checkpoint.pth.tar'):
        '''
        Loading checkpoints.
        If specified the best to load,
        the function will load the checkpoint tagged best.
        '''

        if best and 'best.pth.tar' in os.listdir('.'):
            self.network.load_state_dict(
                torch.load(self.checkpoint_path + '/' + 'best.pth.tar'))
        else:
            try:
                self.network.load_state_dict(
                    torch.load(self.checkpoint_path + '/' + filename))
            except Exception:
                self.command_log(
                    f'No such file {self.checkpoint_path+"/"+filename}...',
                    True)
                raise Exception

    def command_log(self, msg, error=False):
        '''
        Logging info in command line style
        '''
        if error:
            print("[Err] {}".format(msg))
        else:
            print("[Msg] {}".format(msg))
