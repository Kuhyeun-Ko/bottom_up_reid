from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, fixed_layer=True):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.fixed_layer = fixed_layer

    def train(self, epoch, data_loader, optimizer, all_label_to_clusterid, print_freq=1):
        self.model.train()

        if self.fixed_layer:
            # The following code is used to keep the BN on the first three block fixed 
            fixed_bns = []
            for idx, (name, module) in enumerate(self.model.module.named_modules()):
                if name.find("layer3") != -1:
                    assert len(fixed_bns) == 22
                    break
                if name.find("bn") != -1:
                    fixed_bns.append(name)
                    module.eval() 

        losses=[]
        for i, inputs in enumerate(data_loader):
            
            # training
            inputs, targets, indexs, sceneid, label_to_pairs = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets, indexs, sceneid, label_to_pairs, all_label_to_clusterid)
            if loss!=0: losses.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % print_freq == 0:
            # if (i %100) == 0:
                if len(losses)!=0:avg_loss=sum(losses)/len(losses)
                else: avg_loss=0
                print('Epoch: [%d][%d/%d] / Avg. losses: %.4f'%(epoch, i + 1, len(data_loader), avg_loss))
                losses=[]

                if self.criterion.num_pos!=0: pos_ratio=(float(self.criterion.num_hpos)/float(self.criterion.num_pos))
                else: pos_ratio=0
                if self.criterion.num_neg!=0: neg_ratio=(float(self.criterion.num_hneg)/float(self.criterion.num_neg))
                else: neg_ratio=0
                print('hard pos(%d)/pos((%d)%d): %.2f / hard neg(%d)/neg((%d)%d): %.2f'%(self.criterion.num_hpos,self.criterion.num_pos_notable, self.criterion.num_pos,pos_ratio,self.criterion.num_hneg, self.criterion.num_neg_notable, self.criterion.num_neg, neg_ratio))
                
                self.criterion.num_pos=0
                self.criterion.num_pos_notable=0
                self.criterion.num_hpos=0
                self.criterion.num_neg=0
                self.criterion.num_neg_notable=0
                self.criterion.num_hneg=0



    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids, indexs, videoid, sceneid, label_to_pairs = inputs
        inputs = Variable(imgs, requires_grad=False)
        targets = Variable(videoid.cuda())
        return inputs, targets, indexs, sceneid, label_to_pairs

    def _forward(self, inputs, targets, indexs, sceneid, label_to_pairs, all_label_to_clusterid):
        # output is feature
        outputs, _ = self.model(inputs)
        # output is similarity
        loss, outputs = self.criterion(outputs, targets, indexs, label_to_pairs, all_label_to_clusterid)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]
        return loss, prec

