from __future__ import absolute_import

import torch
import torch.nn.functional as F
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

    def forward(self, inputs, targets, label_to_pairs, indexs):
        outputs = Exclusive(self.V)(inputs, targets) * self.t
        bu_loss = F.cross_entropy(outputs, targets, weight=self.weight)

        # hard negative mining
        h_loss=[]
        normalized_inputs=F.normalize(inputs, dim=1)
        sim=normalized_inputs.mm(normalized_inputs.t())
        for i in range(len(sim)):

            # hard positive
            psims_idx=[]
            for pidx in label_to_pairs[i][0]: psims_idx.append((indexs==pidx).nonzero()[0])
            psims=sim[i,psims_idx]
            hpsims=torch.tensor([torch.min(psims)]).cuda()
            thrd=hpsims.clone()
            
            # hard negative
            nsims_idx=[]
            for nidx in label_to_pairs[i][1]: nsims_idx.append((indexs==nidx).nonzero()[0])
            nsims=sim[i,nsims_idx]
            thrd=torch.tensor(thrd-0.3).cuda()
            hnsims=nsims[thrd<nsims]
            
            # calculate loss
            hp_loss=F.binary_cross_entropy_with_logits(hpsims, torch.ones(hpsims.shape).cuda())
            if hnsims.shape[0]==0: hn_loss=torch.tensor([0]).cuda()
            else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())
            h_loss_=hp_loss+hn_loss
            h_loss.append(h_loss_)

        h_loss= sum(h_loss)/len(h_loss)

        # hard negative mining with table self.V
        loss=bu_loss+h_loss

        return loss, outputs