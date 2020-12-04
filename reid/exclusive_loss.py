from __future__ import absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, autograd


class Exclusive(autograd.Function):
    def __init__(self, V):
        super(Exclusive, self).__init__()
        self.V = V

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.V.t())
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            self.V[y] = F.normalize( (self.V[y] + x) / 2, p=2, dim=0)
        return grad_inputs, None


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0,
                 weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        
        self.use_prior=False
        self.w_bu=1.
        self.w_h=1.
        self.w_th=3.
        self.p_margin=0.2
        self.n_margin=0.3
        print('use_prior: %s, w_bu: %.2f, w_h: %.2f, w_th: %.2f, p_margin: %.2f, n_margin: %.2f'%(self.use_prior, self.w_bu, self.w_h, self.w_th, self.p_margin, self.n_margin))

    def forward(self, inputs, targets, label_to_pairs, indexs):

        outputs = Exclusive(self.V)(inputs, targets) * self.t
        bu_loss = F.cross_entropy(outputs, targets, weight=self.weight)

        if self.use_prior:
            h_loss = self.forward_hard_negative_mining(inputs, label_to_pairs, indexs)
            th_loss = self.forward_hard_negative_mining_with_table(inputs, targets, label_to_pairs, indexs)
        else:
            h_loss = self.hard_negative_mining(inputs, targets, label_to_pairs, indexs)
            th_loss = self.hard_negative_mining_with_table(inputs, targets, label_to_pairs, indexs)

        # loss=bu_loss
        # loss=bu_loss+h_loss
        loss=self.w_bu*bu_loss+self.w_h*h_loss+self.w_th*th_loss

        return loss, outputs


    ## hard negative mining
    def forward_hard_negative_mining(self, inputs, label_to_pairs, indexs):

        h_loss=[]
        normalized_inputs=F.normalize(inputs, dim=1)
        sims=normalized_inputs.mm(normalized_inputs.t())

        # hard positive
        psims=2*torch.ones(sims.shape).cuda()
        for i, pairs in enumerate(label_to_pairs): 
            for ppair in pairs[0]:
                if len((ppair==indexs).nonzero())!=0: psims[i, (ppair==indexs).nonzero().item()]=sims[i, (ppair==indexs).nonzero().item()]
        # threshold
        thd_psims=psims.clone()
        n_thrds=torch.min(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
        n_thrds-=self.n_margin
        thd_psims[thd_psims==2]=-2
        p_thrds=torch.max(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
        p_thrds-=self.p_margin
        hpsims=psims[psims<p_thrds]

        if hpsims.shape[0]==0: hp_loss=torch.zeros(1).cuda()
        else: hp_loss=F.binary_cross_entropy_with_logits(hpsims, torch.ones(hpsims.shape).cuda())

        # hard negative
        nsims=-2*torch.ones(sims.shape).cuda()
        for i, pairs in enumerate(label_to_pairs): 
            for npair in pairs[1]:
                if len((npair==indexs).nonzero())!=0: nsims[i, (npair==indexs).nonzero().item()]=sims[i, (npair==indexs).nonzero().item()]
        hnsims=nsims[nsims>n_thrds]

        if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
        else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

        # loss calculate ovelapped
        h_loss=hp_loss+hn_loss

        return h_loss

    ## hard negative mining with table self.V
    def forward_hard_negative_mining_with_table(self, inputs, targets, label_to_pairs, indexs):
        
        # When table self.V is filled
        if len((torch.sum(self.V, 1)==torch.zeros(torch.sum(self.V, 1).shape).cuda()).nonzero())==0:

            # threshold
            normalized_inputs=F.normalize(inputs, dim=1)
            n_thrds=torch.zeros(targets.shape)
            n_thrds=normalized_inputs.mm(self.V.t())[torch.arange(len(targets)).cuda(), targets].unsqueeze(1).repeat(1, self.V.shape[0])
            n_thrds-=self.n_margin
            
            # hard negative
            nsims_=self.V[targets].mm(self.V.t()) 
            nsims=-2*torch.ones(nsims_.shape).cuda()
            for i, pairs in enumerate(label_to_pairs): 
                for npair in pairs[1]:
                    if len((npair==indexs).nonzero())!=0: nsims[i, (npair==indexs).nonzero().item()]=nsims_[i, (npair==indexs).nonzero().item()]
            hnsims=nsims[(n_thrds<nsims)&(nsims<0.9999)]

            if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
            else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

            # loss calculate ovelapped
            th_loss=hn_loss
        
        else: th_loss=torch.zeros(1).cuda()

        return th_loss


    ## hard negative mining without prior
    def hard_negative_mining(self, inputs, targets, label_to_pairs, indexs):

        h_loss=[]
        normalized_inputs=F.normalize(inputs, dim=1)
        sims=normalized_inputs.mm(normalized_inputs.t())

        # hard positive
        psims=2*torch.ones(sims.shape).cuda()
        for i, (target, sim) in enumerate(zip(targets, sims)): psims[i,target==targets]=sim[target==targets]
        psims=psims[~torch.eye(psims.shape[0]).type(torch.bool)].reshape(psims.shape[0], -1)

        # threshold
        thd_psims=psims.clone()
        n_thrds=torch.min(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
        n_thrds-=self.n_margin
        thd_psims[thd_psims==2]=-2
        p_thrds=torch.max(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
        p_thrds-=self.p_margin
        hpsims=psims[psims<p_thrds]

        if hpsims.shape[0]==0: hp_loss=torch.zeros(1).cuda()
        else: hp_loss=F.binary_cross_entropy_with_logits(hpsims, torch.ones(hpsims.shape).cuda())

        # hard negative
        nsims=-2*torch.ones(sims.shape).cuda()
        for i, (target, sim) in enumerate(zip(targets, sims)): nsims[i,target!=targets]=sim[target!=targets]
        nsims=nsims[~torch.eye(nsims.shape[0]).type(torch.bool)].reshape(nsims.shape[0], -1)
        hnsims=nsims[nsims>n_thrds]

        if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
        else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

        # loss calculate ovelapped
        h_loss=hp_loss+hn_loss

        return h_loss

    ## hard negative mining with table self.V no prior
    def hard_negative_mining_with_table(self, inputs, targets, label_to_pairs, indexs):
        
        # When table self.V is filled
        if len((torch.sum(self.V, 1)==torch.zeros(torch.sum(self.V, 1).shape).cuda()).nonzero())==0:

            # threshold
            normalized_inputs=F.normalize(inputs, dim=1)
            n_thrds=torch.zeros(targets.shape)
            n_thrds=normalized_inputs.mm(self.V.t())[torch.arange(len(targets)).cuda(), targets].unsqueeze(1).repeat(1, self.V.shape[0])
            n_thrds-=self.n_margin
            
            # hard negative
            nsims=self.V[targets].mm(self.V.t()) 
            hnsims=nsims[(n_thrds<nsims)&(nsims<0.9999)]

            if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
            else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

            # loss calculate ovelapped
            th_loss=hn_loss
        
        else: th_loss=torch.zeros(1).cuda()

        return th_loss

