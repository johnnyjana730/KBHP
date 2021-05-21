import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from sklearn.metrics import f1_score, roc_auc_score
import model.manifolds as manifolds
from model.hyperplan.hyper_base import *



class hpNet_rela_bi_pur(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(hpNet_rela_bi_pur, self).__init__()
        self._parse_args(args, n_entity, n_relation)

        self.entity_emb_matrix_sw = nn.Embedding(self.n_entity, self.dim)
        self.entity_emb_matrix_sw.weight.data = self.init_size * torch.randn((self.n_entity, self.dim))


        self.rela_plus_emb_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # torch.nn.init.xavier_uniform_(self.rela_plus_emb_matrix.weight)
        self.rela_plus_emb_matrix.weight.data = torch.randn((self.n_relation + 1, self.dim))

        if self.manifold.name == 'Hyperboloid':
            self.relation_emb_matrix = nn.Embedding(self.n_relation + 1, (self.dim + 1) * (self.dim + 1))
            # torch.nn.init.xavier_uniform_(self.relation_emb_matrix.weight)
            self.relation_emb_matrix.weight.data = torch.randn(self.n_relation + 1, (self.dim + 1) * (self.dim + 1))
        else:
            self.relation_emb_matrix = nn.Embedding(self.n_relation + 1, self.dim * self.dim)
            # torch.nn.init.xavier_uniform_(self.relation_emb_matrix.weight)
            self.relation_emb_matrix.weight.data =  torch.randn(self.n_relation + 1, self.dim * self.dim)

        self.criterion = nn.BCELoss()

        self.dropout = 0.5
        self.dc = FermiDiracDecoder(args, r=args.r, t=args.t)

        # self.rel_diag_matrix = nn.Embedding(self.n_relation + 1, self.dim * self.dim)
        # self.rel_diag_matrix.weight.data = 2 * torch.rand((self.n_relation + 1,  self.dim * self.dim)) - 1.0

        self.one_like_bahop =  nn.Parameter(torch.ones_like(torch.randn(args.batch_size, args.n_hop)).double(), requires_grad = False)

        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}

        self.att_beta = args.att_beta



        # self.rel_diag_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # self.rel_diag_matrix.weight.data = 2 * torch.rand((self.n_relation + 1,  self.dim)) - 1.0

        # self.rel_diag_2_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # self.rel_diag_2_matrix.weight.data = 2 * torch.rand((self.n_relation + 1, self.dim)) - 1.0

        # self.context_vec = nn.Embedding(self.n_relation + 1,  self.dim)
        # self.context_vec.weight.data = self.init_size * torch.randn((self.n_relation + 1, self.dim))

        # self.scale = nn.Parameter(torch.Tensor([1. / np.sqrt(self.dim)]),requires_grad = False)

        # self.criterion = nn.BCELoss()

        # self.dropout = 0.5
        # self.dc = FermiDiracDecoder(r=args.r, t=args.t)
    
        # self.one_like_bahop =  nn.Parameter(torch.ones_like(torch.randn(args.batch_size, args.n_hop)), requires_grad = False)
        # self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        # self.att_beta = args.att_beta

    def _parse_args(self, args, n_entity, n_relation):

        self.init_size = 1e-2
        self.n_entity = n_entity
        self.n_relation = n_relation
        self.dim = args.dim
        self.n_hop = args.n_hop
        self.kge_weight = args.kge_weight
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        self.n_memory = args.n_memory
        self.item_update_mode = args.item_update_mode
        self.using_all_hops = args.using_all_hops
        self.debug_test = args.debug_test

        self.manifold = getattr(manifolds, args.manifold_name)()

        self.uemb_generate = self._key_hper_addressing
        self.fin_predict = self.hyper_predict     

    # def forward(self, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):
    def forward(self, users: torch.LongTensor, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):

        self.item_embeddings = self.entity_emb_matrix_sw(items)

        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(self.item_embeddings)
            self.item_embeddings = torch.cat([o[:, 0:1], self.item_embeddings], dim=-1)

        self.h_emb_list = []
        self.t_emb_list = []
        self.r_emb_list = []
        self.r_plus_emb_list = []
        self.cur = []
        for i in range(self.n_hop):
            h_emb = self.entity_emb_matrix_sw(memories_h[i])
            t_emb = self.entity_emb_matrix_sw(memories_t[i])
            r_plus_emb = self.rela_plus_emb_matrix(memories_r[i])

            self.cur.append(1.0)
            if self.manifold.name == 'Hyperboloid':
                o = torch.zeros_like(h_emb)
                h_emb = torch.cat([o[:, :, 0:1], h_emb], dim=-1)
                t_emb = torch.cat([o[:, :, 0:1], t_emb], dim=-1)
                r_plus_emb = torch.cat([o[:, :, 0:1], r_plus_emb], dim=-1)
                r_emb = self.relation_emb_matrix(memories_r[i]).view(-1, self.n_memory, self.dim + 1, self.dim + 1)
            else:
                r_emb = self.relation_emb_matrix(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim)

            # [batch size, n_memory, dim]
            self.h_emb_list.append(h_emb)
            # [batch size, n_memory, dim]
            self.t_emb_list.append(t_emb)
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(r_emb/10)
            # [batch size, n_memory, dim]
            self.r_plus_emb_list.append(r_plus_emb)
            # self.uemb_generate = self._key_hper_addressing
            # self.fin_predict = self.hyper_predict

        self.pre_diag_emb = self.relation_emb_matrix(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])).cuda())

        o_list = self.uemb_generate()
        scores = self.fin_predict(self.item_embeddings, o_list).squeeze()

        return_dict = self._compute_loss(scores, labels.double())
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels):

        base_loss = self.criterion(scores, labels)

        kge_loss = 0
        # for hop in range(self.n_hop):
        #     # [batch size, n_memory, 1, dim]
        #     h_expanded = torch.unsqueeze(self.h_emb_list[hop], dim=2)
        #     # [batch size, n_memory, dim, 1]
        #     t_expanded = torch.unsqueeze(self.t_emb_list[hop], dim=3)
        #     # [batch size, n_memory, dim, dim]
        #     hRt = torch.squeeze(
        #         torch.matmul(torch.matmul(h_expanded, self.r_emb_list[hop]), t_expanded)
        #     )
        #     kge_loss += torch.sigmoid(hRt).mean()
        # kge_loss = -self.kge_weight * kge_loss

        for hop in range(self.n_hop):

            h_emb_sp = self.h_emb_list[hop].shape
            batch_men, men, dim = h_emb_sp[0], h_emb_sp[1], h_emb_sp[2]

            # print('dim = ', dim)

            # print('self.h_emb_list[hop] = ', self.h_emb_list[hop].shape)
            # print('self.r_emb_list[hop] = ', self.r_emb_list[hop].shape)
            # print('self.t_emb_list[hop] = ', self.t_emb_list[hop].shape)

            r_emb_dpwt = self.r_emb_list[hop]
            hyp_h_emb = self.h_emb_list[hop].view(-1, dim, 1)
            Rh = torch.matmul(r_emb_dpwt.view(-1,dim, dim), hyp_h_emb).view(-1, dim)
            Rh = self.manifold.proj_tan0_exp(Rh, self.cur[0])

            hyp_t_emb = self.t_emb_list[hop].view(-1, dim)
            Ri = self.manifold.proj_tan0_exp(hyp_t_emb.view(-1, dim), self.cur[0]) 
            hhRt = self.dc.forward(self.manifold.sqdist(Rh, Ri, self.cur[0]))
            kge_loss += hhRt.mean()

        kge_loss = - self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (self.h_emb_list[hop] * self.h_emb_list[hop]).sum()
            l2_loss += (self.t_emb_list[hop] * self.t_emb_list[hop]).sum()
            l2_loss += (self.r_emb_list[hop] * self.r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        
        return dict(base_loss=base_loss, kge_loss=0, l2_loss=0, loss=loss)

    def _key_hper_addressing(self):
        kle_o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_emb_sp = self.h_emb_list[hop].shape
            batch_men, men, dim = h_emb_sp[0], h_emb_sp[1], h_emb_sp[2]

            r_emb_dpwt = F.dropout(self.r_emb_list[hop], self.dropout, training=self.training)
            Rh = torch.matmul(r_emb_dpwt.view(-1,dim, dim), self.h_emb_list[hop].view(-1, dim, 1)).view(-1, dim)
            Rh = self.manifold.proj_tan0_exp(Rh, self.cur[0]) 

            # print('Rh = ', Rh.shape)

            item_emb = (self.item_embeddings.unsqueeze(1)).repeat(1,men,1)

            # print('item_emb = ', item_emb.shape)
            # input()
            # Ri = torch.matmul(r_emb_dpwt.view(-1,dim, dim), item_emb.view(-1, dim, 1)).view(-1, dim)
            Ri = self.manifold.proj_tan0_exp(item_emb.view(-1, dim), self.cur[0]) 

            # print('Ri = ', Ri.type())

            probs = self.manifold.sqdist(Rh, Ri, self.cur[0])

            # print('probs = ', probs.type())

            probs = torch.exp(- self.att_beta * probs + 0.05)
            hyp_t_emb = self.manifold.proj_tan0_exp(self.t_emb_list[hop].view(-1, dim), self.cur[0])
            kle_t_emb = self.manifold.to_klein(hyp_t_emb, c=self.cur[0])

            probs = self.manifold.lorentz_factor(kle_t_emb, c=self.cur[0]) * probs.view(-1,1)
            probs = probs.view(-1, men)
            probs_expanded = (probs/(probs.sum(1).unsqueeze(1))).unsqueeze(-1)

            if self.manifold.name == 'Hyperboloid': kle_t_emb = kle_t_emb.view(-1,men, dim-1)
            else:  kle_t_emb = kle_t_emb.view(-1,men, dim)

            kle_o = ((kle_t_emb * probs_expanded).sum(dim=1)).unsqueeze(1)

            kle_o_list.append(kle_o)

        return kle_o_list

    def hyper_predict(self, item_embeddings, kle_o_list):

        # v = torch.ones_like(torch.rand(2, len(o_list))
        kle_o_list = torch.cat(kle_o_list, dim=1)
        # print('kle_o_list = ', kle_o_list)
        # input()
        kle_shape = kle_o_list.shape
        probs = self.one_like_bahop[:kle_shape[0],:]
        probs = self.manifold.lorentz_factor(kle_o_list.view(-1,kle_shape[-1]), c=self.cur[0]) * probs.view(-1,1)
        
        # print('probs = ', probs.type())

        probs = probs.view(kle_shape[0], kle_shape[1])
        probs_expanded = (probs/(probs.sum(1).unsqueeze(1))).unsqueeze(-1)

        kle_o = (kle_o_list * probs_expanded).sum(dim=1)
        hyp_o = self.manifold.klein_to(kle_o, self.cur[0])
        hyp_o = self.manifold.proj(hyp_o, self.cur[0])

        dim = item_embeddings.shape[-1]

        # item_emb = (item_embeddings.unsqueeze(1)).repeat(1,men,1)
        pre_diag_emb = F.dropout(self.pre_diag_emb, self.dropout, training=self.training)
        Ri = torch.matmul(pre_diag_emb.view(-1,dim, dim), item_embeddings.view(-1, dim, 1)).view(-1, dim)
        item_ri = self.manifold.proj_tan0_exp(Ri, self.cur[0]) 

        return self.dc.forward(self.manifold.sqdist(hyp_o, item_ri, self.cur[0]))


    def _update_item_embedding(self, item_embeddings, o):
        if self.item_update_mode == "replace":
            item_embeddings = o
        elif self.item_update_mode == "plus":
            item_embeddings = item_embeddings + o
        elif self.item_update_mode == "replace_transform":
            item_embeddings = self.transform_matrix(o)
        elif self.item_update_mode == "plus_transform":
            item_embeddings = self.transform_matrix(item_embeddings + o)
        else:
            raise Exception("Unknown item updating mode: " + self.item_update_mode)
        return item_embeddings



    def evaluate(self, user, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(user, items, labels, memories_h, memories_r, memories_t)
    # def evaluate(self, items, labels, memories_h, memories_r, memories_t):
        # return_dict = self.forward(items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].cpu().detach().numpy() 
        labels = labels.cpu().detach().numpy() 
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc, f1

