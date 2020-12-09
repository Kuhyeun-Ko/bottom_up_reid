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
        
        self.use_prior=True
        self.w_bu=0.
        self.w_h=0.
        self.w_th=1.
        self.p_margin=0.3
        self.n_margin=0.3

        self.num_pos=0
        self.num_pos_notable=0
        self.num_hpos=0
        self.num_neg=0
        self.num_neg_notable=0
        self.num_hneg=0
        self.num_tneg=0
        self.num_thneg=0
        print('use_prior: %s, w_bu: %.2f, w_h: %.2f, w_th: %.2f, p_margin: %.2f, n_margin: %.2f'%(self.use_prior, self.w_bu, self.w_h, self.w_th, self.p_margin, self.n_margin))

    def forward(self, inputs, targets, label_to_pairs, indexs, all_label_to_clusterid):

        outputs = Exclusive(self.V)(inputs, targets) * self.t
        bu_loss = F.cross_entropy(outputs, targets, weight=self.weight)

        if self.use_prior:
            h_loss = self.forward_hard_negative_mining(inputs, targets, label_to_pairs, indexs, all_label_to_clusterid)
            th_loss = self.forward_hard_negative_mining_with_table(inputs, targets, label_to_pairs, indexs, all_label_to_clusterid)
        else:
            h_loss = self.hard_negative_mining(inputs, targets, label_to_pairs, indexs)
            th_loss = self.hard_negative_mining_with_table(inputs, targets, label_to_pairs, indexs)
            raise ValueError

        # loss=bu_loss
        # loss=bu_loss+h_loss
        loss=self.w_bu*bu_loss+self.w_h*h_loss+self.w_th*th_loss

        return loss, outputs


    ## hard negative mining
    def forward_hard_negative_mining(self, inputs, targets, label_to_pairs, indexs, all_label_to_clusterid):

        h_loss=[]
        normalized_inputs=F.normalize(inputs, dim=1)
        sims=normalized_inputs.mm(normalized_inputs.t())
        tsims=normalized_inputs.mm(self.V.t())

        # positive
        psims=[[] for i in range(sims.shape[0])]
        for i, (pairs, target) in enumerate(zip(label_to_pairs, targets)): 
            if tsims[i, target] !=0: psims[i].append(tsims[i, target])
            for ppair in pairs[0]: 
                if len((ppair==indexs).nonzero())!=0:  
                    for index in [(ppair==indexs).nonzero().item()]: 
                        psims[i].append(sims[i, index])
                        self.num_pos_notable+=1
       
        # negative
        nsims=[[] for i in range(sims.shape[0])]
        for i, pairs in enumerate(label_to_pairs): 
            for npair in pairs[1]:
                if tsims[i, all_label_to_clusterid[npair]] !=0: nsims[i].append(tsims[i, all_label_to_clusterid[npair]])
                if len((npair==indexs).nonzero())!=0: 
                    for index in [(npair==indexs).nonzero().item()]: 
                        nsims[i].append(sims[i, index])
                        self.num_neg_notable+=1
      
        # threshold    
        p_thrds=[ max(nsim or [-3]) for nsim in nsims]
        p_thrds=list(map(lambda x: x+self.p_margin, p_thrds))
        n_thrds=[ min(psim or [3]) for psim in psims]
        n_thrds=list(map(lambda x: x-self.n_margin, n_thrds))
        
        # hard positive and hard negatvie
        hpsims=[ list(filter(lambda x: x<p_thrds[i], psim)) for i, psim in enumerate(psims)]
        hpsims=sum(hpsims, [])
        hpsims=torch.tensor(hpsims).cuda()

        hnsims=[ list(filter(lambda x: ((x>n_thrds[i])& (x<0.999999)), nsim)) for i, nsim in enumerate(nsims)]
        hnsims=sum(hnsims, [])
        hnsims=torch.tensor(hnsims).cuda()

        if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
        else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

        if len(hpsims)==0: hp_loss=torch.zeros(1).cuda()
        else: hp_loss=F.binary_cross_entropy_with_logits(hpsims, torch.ones(hpsims.shape).cuda())
        
        
        # loss calculate ovelapped
        h_loss=hp_loss+hn_loss

        self.num_pos+=len(sum(psims, []))
        self.num_hpos+=hpsims.shape[0]
        self.num_neg+=len(sum(nsims, []))
        self.num_hneg+=hnsims.shape[0]

        return h_loss

    ## hard negative mining with table self.V
    def forward_hard_negative_mining_with_table(self, inputs, targets, label_to_pairs, indexs, all_label_to_clusterid):

        # When table self.V is filled
        if len((torch.sum(self.V, 1)==torch.zeros(torch.sum(self.V, 1).shape).cuda()).nonzero())==0:

            # threshold
            normalized_inputs=F.normalize(inputs, dim=1)
            sims=normalized_inputs.mm(self.V.t())
            n_thrds=[]
            for i, target in enumerate(targets): n_thrds.append(sims[i, target])
            n_thrds=list(map(lambda x: x-self.n_margin, n_thrds))
            assert len(targets)==len(n_thrds), "n_thrds has wrong length."
            
            # hard negative
            tsims=self.V[targets].mm(self.V.t()) 
            nsims=[[] for i in range(tsims.shape[0])]
            for i, pairs in enumerate(label_to_pairs): 
                all_label_to_clusterid_=list(set([ clusterid for i, clusterid in enumerate(all_label_to_clusterid) if i in pairs[1]]))
                for clusterid_ in all_label_to_clusterid_: 
                    nsims[i].append(tsims[i, clusterid_])
            hnsims=[ list(filter(lambda x: ((x>n_thrds[i]) & (x<0.999999)), nsim)) for i, nsim in enumerate(nsims)]
            hnsims=sum(hnsims, [])
            hnsims=torch.tensor(hnsims).cuda()

            if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
            else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

            # loss calculate ovelapped
            th_loss=hn_loss

            self.num_tneg+=len(sum(nsims, []))
            self.num_thneg+=hnsims.shape[0]

        else: th_loss=torch.zeros(1).cuda()

        return th_loss


    # ## hard negative mining without prior
    # def hard_negative_mining(self, inputs, targets, label_to_pairs, indexs):

    #     h_loss=[]
    #     normalized_inputs=F.normalize(inputs, dim=1)
    #     sims=normalized_inputs.mm(normalized_inputs.t())

    #     # hard positive
    #     psims=10*torch.ones(sims.shape).cuda()
    #     for i, (target, sim) in enumerate(zip(targets, sims)): 
    #         psims[i,target==targets]=sim[target==targets]
    #         if normalized_inputs.mm(self.V.t())[i, target] !=0: psims[i, i]= normalized_inputs.mm(self.V.t())[i, target]
    #     # psims=psims[~torch.eye(psims.shape[0]).type(torch.bool)].reshape(psims.shape[0], -1)

    #     # threshold
    #     thd_psims=psims.clone()
    #     n_thrds=torch.min(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
    #     n_thrds-=self.n_margin
    #     thd_psims[thd_psims==10]=-10
    #     p_thrds=torch.max(thd_psims, dim=1, keepdim=True).values.repeat(1, thd_psims.shape[1])
    #     p_thrds-=self.p_margin
    #     hpsims=psims[psims<p_thrds]
    #     if hpsims.shape[0]==0: hp_loss=torch.zeros(1).cuda()
    #     else: hp_loss=F.binary_cross_entropy_with_logits(hpsims, torch.ones(hpsims.shape).cuda())

    #     # hard negative
    #     nsims=-10*torch.ones(sims.shape).cuda()
    #     for i, (target, sim) in enumerate(zip(targets, sims)): 
    #         nsims[i,target!=targets]=sim[target!=targets]
    #         # if normalized_inputs.mm(self.V.t())[i, target] !=0: nsims[i, i]= normalized_inputs.mm(self.V.t())[i, target]
    #     # nsims=nsims[~torch.eye(nsims.shape[0]).type(torch.bool)].reshape(nsims.shape[0], -1)
    #     hnsims=nsims[nsims>n_thrds]

    #     if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
    #     else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

    #     # loss calculate ovelapped
    #     h_loss=hp_loss+hn_loss

    #     self.num_pos+=len((psims!=10).nonzero())
    #     self.num_hpos+=hpsims.shape[0]
    #     self.num_neg+=len((nsims!=-10).nonzero())
    #     self.num_hneg+=hnsims.shape[0]

    #     return h_loss

    # ## hard negative mining with table self.V no prior
    # def hard_negative_mining_with_table(self, inputs, targets, label_to_pairs, indexs):
        
    #     # When table self.V is filled
    #     if len((torch.sum(self.V, 1)==torch.zeros(torch.sum(self.V, 1).shape).cuda()).nonzero())==0:

    #         # threshold
    #         normalized_inputs=F.normalize(inputs, dim=1)
    #         n_thrds=torch.zeros(targets.shape)
    #         n_thrds=normalized_inputs.mm(self.V.t())[torch.arange(len(targets)).cuda(), targets].unsqueeze(1).repeat(1, self.V.shape[0])
    #         n_thrds-=self.n_margin
            
    #         # hard negative
    #         nsims=self.V[targets].mm(self.V.t()) 
    #         hnsims=nsims[(n_thrds<nsims)&(nsims<0.999999)]

    #         if hnsims.shape[0]==0: hn_loss=torch.zeros(1).cuda()
    #         else: hn_loss=F.binary_cross_entropy_with_logits(hnsims, torch.zeros(hnsims.shape).cuda())

    #         # loss calculate ovelapped
    #         th_loss=hn_loss

    #         self.num_tneg+= nsims.shape[0]*nsims.shape[1]
    #         self.num_thneg+=hnsims.shape[0]
        
    #     else: th_loss=torch.zeros(1).cuda()

    #     return th_loss

