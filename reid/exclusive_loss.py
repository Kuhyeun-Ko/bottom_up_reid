from __future__ import absolute_import

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn, autograd


class Real_negative_exclusive(autograd.Function):
    def __init__(self, V, label_to_pairs, all_label_to_clusterid):
        super(Real_negative_exclusive, self).__init__()
        self.V = V
        self.label_to_pairs=label_to_pairs
        self.all_label_to_clusterid=all_label_to_clusterid

    def forward(self, inputs, targets):
        self.save_for_backward(inputs, targets)
        outputs = inputs.mm(self.V.t())
        for i, pairs in enumerate(self.label_to_pairs):
            mask=torch.ones(len(self.V)).type(torch.bool)
            mask_=list(set([ clusterid for j, clusterid in enumerate(self.all_label_to_clusterid) if j in pairs[1]]))
            mask[mask_]=False
            outputs[i, mask]=0
        return outputs

    def backward(self, grad_outputs):
        inputs, targets = self.saved_tensors
        grad_inputs = grad_outputs.mm(self.V) if self.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            self.V[y] = F.normalize( (self.V[y] + x) / 2, p=2, dim=0)
        return grad_inputs, None

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
    def __init__(self, num_features, num_classes, bottom_up_real_negative=False, ms_table=False, ms_real_negative=False, t=1.0, weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))
        
        self.w_bu=1.
        self.w_ms=0.
        self.bottom_up_real_negative=bottom_up_real_negative
        self.ms_real_negative=ms_real_negative
        self.ms_table=ms_table
        self.p_margin=0.5
        self.n_margin=0.5
        
        self.num_pos=0
        self.num_pos_notable=0
        self.num_hpos=0
        self.num_neg=0
        self.num_neg_notable=0
        self.num_hneg=0
        print('w_bu: %.2f, w_ms: %.2f, p_margin: %.2f, n_margin: %.2f'%(self.w_bu, self.w_ms, self.p_margin, self.n_margin))
        print('bottom_up_real_negative: %s, ms_real_negative: %s, ms_table: %s'%(self.bottom_up_real_negative, self.ms_real_negative, self.ms_table))

    def forward(self, inputs, targets, indexs, label_to_pairs, all_label_to_clusterid):
        
        # bu loss        
        if self.bottom_up_real_negative:
            outputs = Real_negative_exclusive(self.V, label_to_pairs, all_label_to_clusterid)(inputs, targets) * self.t
            bu_loss = F.cross_entropy(outputs, targets, weight=self.weight)
        else:
            outputs = Exclusive(self.V)(inputs, targets) * self.t
            bu_loss = F.cross_entropy(outputs, targets, weight=self.weight)
        
        # ms loss
        if self.ms_real_negative: ms_loss, outputs = self.real_negative_ms(inputs, targets, indexs, label_to_pairs, all_label_to_clusterid)
        else: ms_loss, outputs = self.ms(inputs, targets, indexs, label_to_pairs, all_label_to_clusterid)
        
        # loss=self.w_bu*bu_loss
        # loss=self.w_ms*ms_loss
        loss=self.w_ms*ms_loss+self.w_bu*bu_loss
        print('in')

        return loss, outputs


    ## hard negative mining
    def real_negative_ms(self, inputs, targets, indexs, label_to_pairs, all_label_to_clusterid):

        ms_loss=[]
        sims=inputs.mm(inputs.t())
        # if self.ms_real_negative: tsims=Exclusive(self.V)(inputs, targets, label_to_pairs, all_label_to_clusterid) 
        # else: tsims=no_prior_Exclusive(self.V)(inputs, targets) 
        tsims=Exclusive(self.V)(inputs, targets)

        psims=[[] for i in range(sims.shape[0])]
        nsims=[[] for i in range(sims.shape[0])]
        for i, (pairs, target) in enumerate(zip(label_to_pairs, targets)): 
            
            # positive
            for ppair in pairs[0]: 
                if len((ppair==indexs).nonzero())!=0:  
                    for index in [(ppair==indexs).nonzero().item()]: 
                        psims[i].append(sims[i, index])
                        self.num_pos_notable+=1
            if self.ms_table:
                if tsims[i, target] !=0: psims[i].append(tsims[i, target])
            
            # negative
            for npair in pairs[1]:
                if len((npair==indexs).nonzero())!=0: 
                    for index in [(npair==indexs).nonzero().item()]: 
                        nsims[i].append(sims[i, index])
                        self.num_neg_notable+=1
                if self.ms_table:
                    if tsims[i, all_label_to_clusterid[npair]] !=0: nsims[i].append(tsims[i, all_label_to_clusterid[npair]])
            
    
        # threshold    
        p_thrds=[ max(nsim or [torch.tensor(-3).cuda()]) for nsim in nsims]
        p_thrds=list(map(lambda x: x+self.p_margin, p_thrds))
        n_thrds=[ min(psim or [torch.tensor(3).cuda()]) for psim in psims]
        n_thrds=list(map(lambda x: x-self.n_margin, n_thrds))

        # hard positive and hard negatvie
        hpsims=[ list(filter(lambda x: x<p_thrds[i], psim)) for i, psim in enumerate(psims)]
        hpsims=sum(hpsims, [])

        hnsims=[ list(filter(lambda x: ((x>n_thrds[i])& (x<torch.tensor(0.999999).cuda())), nsim)) for i, nsim in enumerate(nsims)]
        hnsims=sum(hnsims, [])
       
        if len(hpsims)==0: hp_loss=tsims.mean()*torch.zeros(1).cuda()
        else:
            hpsims_=torch.stack(hpsims)
            # hp_loss=F.mse_loss(hpsims_, torch.ones(hpsims_.shape).cuda())
            hp_loss=F.binary_cross_entropy_with_logits(hpsims_, torch.ones(hpsims_.shape).cuda())
            # hp_loss = 1.0 / 2 * torch.log(1 + torch.sum(torch.exp(-2 * (hpsims_ - 0.5))))

        if len(hnsims)==0: hn_loss=tsims.mean()*torch.zeros(1).cuda()
        else:
            hnsims_=torch.stack(hnsims)
            # hn_loss=F.mse_loss(hnsims_, -torch.ones(hnsims_.shape).cuda())
            hn_loss=F.binary_cross_entropy_with_logits(hnsims_, torch.zeros(hnsims_.shape).cuda())
            # hn_loss = 1.0 / 50 * torch.log(1 + torch.sum(torch.exp( 50 * (hnsims_ - 0.5))))
        

        # loss calculate ovelapped
        ms_loss=hp_loss+hn_loss

        self.num_pos+=len(sum(psims, []))
        self.num_hpos+=len(hpsims)
        self.num_neg+=len(sum(nsims, []))
        self.num_hneg+=len(hnsims)

        return ms_loss, self.t*tsims


    ## hard negative mining
    def ms(self, inputs, targets, indexs, label_to_pairs, all_label_to_clusterid):

        ms_loss=[]
        sims=inputs.mm(inputs.t())
        # if self.ms_real_negative: tsims=Exclusive(self.V)(inputs, targets, label_to_pairs) 
        # else: tsims=no_prior_Exclusive(self.V)(inputs, targets) 
        tsims=Exclusive(self.V)(inputs, targets)

        psims=[[] for i in range(sims.shape[0])]
        nsims=[[] for i in range(sims.shape[0])]
        for i, target in enumerate(targets): 

            # positive            
            for index in (target==targets).nonzero().view(-1).tolist():
                psims[i].append(sims[i, index])
                self.num_pos_notable+=1
            if self.ms_table:
                if tsims[i, target] !=0: psims[i].append(tsims[i, target])
            
            # negative
            for index in (target!=targets).nonzero().view(-1).tolist():
                nsims[i].append(sims[i, index])
                self.num_neg_notable+=1
            
            if self.ms_table:
                mask=torch.zeros(len(self.V)).type(torch.bool).cuda()
                mask[target]=True
                tsims[i, mask]=0
                nsims[i].extend(tsims[i, :][tsims[i, :]!=0])
        
        # threshold    
        p_thrds=[ max(nsim or [torch.tensor(-3).cuda()]) for nsim in nsims]
        p_thrds=list(map(lambda x: x+self.p_margin, p_thrds))
        n_thrds=[ min(psim or [torch.tensor(3).cuda()]) for psim in psims]
        n_thrds=list(map(lambda x: x-self.n_margin, n_thrds))

        # hard positive and hard negatvie
        hpsims=[ list(filter(lambda x: x<p_thrds[i], psim)) for i, psim in enumerate(psims)]
        hpsims=sum(hpsims, [])

        hnsims=[ list(filter(lambda x: ((x>n_thrds[i])& (x<torch.tensor(0.999999).cuda())), nsim)) for i, nsim in enumerate(nsims)]
        hnsims=sum(hnsims, [])
       
        if len(hpsims)==0: hp_loss=tsims.mean()*torch.zeros(1).cuda()
        else:
            hpsims_=torch.stack(hpsims)
            # hp_loss=F.mse_loss(hpsims_, torch.ones(hpsims_.shape).cuda())
            hp_loss=F.binary_cross_entropy_with_logits(hpsims_, torch.ones(hpsims_.shape).cuda())
            # hp_loss = 1.0 / 2 * torch.log(1 + torch.sum(torch.exp(-2 * (hpsims_ - 0.5))))

        if len(hnsims)==0: hn_loss=tsims.mean()*torch.zeros(1).cuda()
        else:
            hnsims_=torch.stack(hnsims)
            # hn_loss=F.mse_loss(hnsims_, -torch.ones(hnsims_.shape).cuda())
            hn_loss=F.binary_cross_entropy_with_logits(hnsims_, torch.zeros(hnsims_.shape).cuda())
            # hn_loss = 1.0 / 50 * torch.log(1 + torch.sum(torch.exp( 50 * (hnsims_ - 0.5))))
        

        # loss calculate ovelapped
        ms_loss=hp_loss+hn_loss

        self.num_pos+=len(sum(psims, []))
        self.num_hpos+=len(hpsims)
        self.num_neg+=len(sum(nsims, []))
        self.num_hneg+=len(hnsims)

        return ms_loss, self.t*tsims


    # ## hard negative mining with table self.V
    # def ms_loss_with_table(self, inputs, targets, label_to_pairs, indexs, all_label_to_clusterid):

    #     # When table self.V is filled
    #     if len((torch.sum(self.V, 1)==torch.zeros(torch.sum(self.V, 1).shape).cuda()).nonzero())==0:

    #         # threshold
    #         normalized_inputs=F.normalize(inputs, dim=1)
    #         sims=normalized_inputs.mm(self.V.t())
    #         n_thrds=[]
    #         for i, target in enumerate(targets): n_thrds.append(sims[i, target])
    #         n_thrds=list(map(lambda x: x-self.n_margin, n_thrds))
    #         assert len(targets)==len(n_thrds), "n_thrds has wrong length."
            
    #         # hard negative
    #         tsims=self.V[targets].mm(self.V.t()) 
    #         nsims=[[] for i in range(tsims.shape[0])]
    #         for i, pairs in enumerate(label_to_pairs): 
    #             all_label_to_clusterid_=list(set([ clusterid for i, clusterid in enumerate(all_label_to_clusterid) if i in pairs[1]]))
    #             for clusterid_ in all_label_to_clusterid_: 
    #                 nsims[i].append(tsims[i, clusterid_])
    #         hnsims=[ list(filter(lambda x: ((x>n_thrds[i]) & (x<0.999999)), nsim)) for i, nsim in enumerate(nsims)]
    #         hnsims=sum(hnsims, [])
    #         hnsims=torch.tensor(hnsims).cuda()

    #         if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
    #         else: 
    #             hn_loss=F.mse_loss(hnsims, torch.zeros(hnsims.shape).cuda())
    #             hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

    #         # loss calculate ovelapped
    #         th_loss=hn_loss

    #         self.num_tneg+=len(sum(nsims, []))
    #         self.num_thneg+=hnsims.shape[0]

    #     else: th_loss=torch.zeros(1).cuda()

    #     return th_loss

    ## hard negative mining without prior



