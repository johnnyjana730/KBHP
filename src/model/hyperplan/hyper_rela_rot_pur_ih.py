import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
torch.autograd.set_detect_anomaly(True)
from sklearn.metrics import f1_score, roc_auc_score
import model.manifolds as manifolds
from model.hyperplan.hyper_base import *

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from tsnecuda import TSNE
import seaborn as sns
import pandas as pd  


def givens_rotations(r, x):
    """Givens rotations.

    Args:
        r: torch.Tensor of shape (N x d), rotation parameters
        x: torch.Tensor of shape (N x d), points to rotate

    Returns:
        torch.Tensor os shape (N x d) representing rotation of x by r
    """
    givens = r.view((r.shape[0], -1, 2))
    givens = givens / torch.norm(givens, p=2, dim=-1, keepdim=True)
    x = x.view((r.shape[0], -1, 2))
    x_rot = givens[:, :, 0:1] * x + givens[:, :, 1:] * torch.cat((-x[:, :, 1:], x[:, :, 0:1]), dim=-1)
    return x_rot.view((r.shape[0], -1))


class hpNet_rela_rot_pur_ih(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(hpNet_rela_rot_pur_ih, self).__init__()
        self._parse_args(args, n_entity, n_relation)

        self.entity_emb_matrix_sw = nn.Embedding(self.n_entity, self.dim)
        # torch.nn.init.xavier_uniform_(self.entity_emb_matrix_sw.weight)
        self.entity_emb_matrix_sw.weight.data = self.init_size * torch.randn((self.n_entity, self.dim))

        self.rela_plus_emb_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # torch.nn.init.xavier_uniform_(self.rela_plus_emb_matrix.weight)
        # self.rela_plus_emb_matrix.weight.data =  torch.rand((self.n_relation + 1, self.dim)) 

        if self.manifold.name == 'Hyperboloid':
            self.rela_plus_emb_matrix.weight.data =  torch.rand((self.n_relation + 1, self.dim)) 
        else:
            self.rela_plus_emb_matrix.weight.data =  torch.rand((self.n_relation + 1, self.dim)) 

        if self.manifold.name == 'Hyperboloid':
            self.relation_emb_matrix = nn.Embedding(self.n_relation, (self.dim) * (self.dim))
            # torch.nn.init.xavier_uniform_(self.relation_emb_matrix.weight)
            self.relation_emb_matrix.weight.data =  torch.randn((self.n_relation, (self.dim) * (self.dim)))
        else:
            self.relation_emb_matrix = nn.Embedding(self.n_relation, self.dim * self.dim)
            # torch.nn.init.xavier_uniform_(self.relation_emb_matrix.weight)
            self.relation_emb_matrix.weight.data =  torch.randn((self.n_relation,(self.dim) * (self.dim)))

        self.rel_diag_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # torch.nn.init.xavier_uniform_(self.rel_diag_matrix.weight)
        self.rel_diag_matrix.weight.data = 2 * torch.rand((self.n_relation + 1,  self.dim)) - 1.0

        self.rel_diag_2_matrix = nn.Embedding(self.n_relation + 1, self.dim)
        # torch.nn.init.xavier_uniform_(self.rel_diag_2_matrix.weight)
        self.rel_diag_2_matrix.weight.data = 2 * torch.rand((self.n_relation + 1, self.dim)) - 1.0

        # self.context_vec = nn.Embedding(self.n_relation + 1,  self.dim)
        # # torch.nn.init.xavier_uniform_(self.context_vec.weight)
        # self.context_vec.weight.data =  torch.randn((self.n_relation + 1, self.dim))
        
        self.context_vec = nn.Embedding(self.n_relation + 1,  self.dim)
        if self.manifold.name == 'Hyperboloid':
            self.context_vec.weight.data =  torch.randn((self.n_relation + 1, self.dim))
        else:
            self.context_vec.weight.data = self.init_size * torch.randn((self.n_relation + 1, self.dim))

        self.scale = nn.Parameter(torch.Tensor([1. / np.sqrt(self.dim)]),requires_grad = False)

        self.criterion = nn.BCELoss()

        self.dropout = 0.5
        self.dc = FermiDiracDecoder(args, r=args.r, t=args.t)
    
        self.one_like_bahop =  nn.Parameter(torch.ones_like(torch.randn(args.batch_size, args.n_hop)), requires_grad = False)
        self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
        self.att_beta = args.att_beta

    def _parse_args(self, args, n_entity, n_relation):

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

        if self.manifold.name == 'Hyperboloid':
            self.init_size = 1e-3
        else:
            self.init_size = 1e-4

    # def forward(self, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):
    def forward(self, users: torch.LongTensor, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):

        # [batch size, dim]
        self.item_embeddings = self.entity_emb_matrix_sw(items)

        self.h_emb_list = []
        self.t_emb_list = []
        self.r_emb_list = []
        self.r_diag_list = []
        self.r_diag_2_list = []
        self.r_daig_att = []
        self.r_plus_emb_list = []

        self.cur = []
        for i in range(self.n_hop):
            h_emb = self.entity_emb_matrix_sw(memories_h[i])
            t_emb = self.entity_emb_matrix_sw(memories_t[i])
            r_plus_emb = self.rela_plus_emb_matrix(memories_r[i])
            r_diag_emb = self.rel_diag_matrix(memories_r[i])
            r_diag_2_emb = self.rel_diag_2_matrix(memories_r[i])
            r_diag_att = self.context_vec(memories_r[i])

            self.cur.append(1.0)
            if self.manifold.name == 'Hyperboloid':
                r_emb = self.relation_emb_matrix(memories_r[i]).view(-1, self.n_memory, self.dim,  self.dim)
            else:
                r_emb = self.relation_emb_matrix(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim)

            # print('r_emb = ', r_emb.shape)
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
            self.r_diag_list.append(r_diag_emb)
            self.r_diag_2_list.append(r_diag_2_emb)
            self.r_daig_att.append(r_diag_att)


        # self.pre_diag_emb = self.rel_diag_matrix(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])))
        # self.pre_diag_2_emb = self.rel_diag_2_matrix(torch.LongTensor([self.n_relation] *int(memories_h[i].shape[0])))
        # self.pre_plus_emb = self.rela_plus_emb_matrix(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])))
        # self.pre_daig_att = self.context_vec(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])))

        self.pre_diag_emb = self.rel_diag_matrix(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])).cuda())
        self.pre_diag_2_emb = self.rel_diag_2_matrix(torch.LongTensor([self.n_relation] *int(memories_h[i].shape[0])).cuda())
        self.pre_plus_emb = self.rela_plus_emb_matrix(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])).cuda())
        self.pre_daig_att = self.context_vec(torch.LongTensor([self.n_relation] * int(memories_h[i].shape[0])).cuda())

        o_list = self.uemb_generate()
        scores = self.fin_predict(self.item_embeddings, o_list).squeeze()

        return_dict = self._compute_loss(scores, labels)
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels):

        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        # for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
        for hop in range(self.n_hop):
            h_emb_sp = self.h_emb_list[hop].shape
            batch_men, men, dim = h_emb_sp[0], h_emb_sp[1], h_emb_sp[2]

            hyp_h_emb = self.manifold.proj_tan0(self.h_emb_list[hop].view(-1, dim), self.cur[0])

            h_emb_rot_q = givens_rotations(self.r_diag_list[hop].view(-1, dim), hyp_h_emb.view(-1, dim)).view((-1, 1, dim))
            h_emb_ref_q = givens_rotations(self.r_diag_2_list[hop].view(-1, dim), hyp_h_emb.view(-1, dim)).view((-1, 1, dim))

            cands = torch.cat([h_emb_rot_q, h_emb_ref_q], dim=1)
            att_weights = torch.sum(self.r_daig_att[hop].view(-1, 1, dim) * cands * self.scale, dim=-1, keepdim=True)
            att_weights = F.softmax(att_weights, dim=1)

            att_q = torch.sum(att_weights * cands, dim=1)
            lhs = self.manifold.proj_tan0_exp(att_q, self.cur[0])
            hyp_r_pl_emb = self.manifold.proj_tan0_exp(self.r_plus_emb_list[hop].view(-1, dim), self.cur[0])
            res = self.manifold.mobius_add(lhs, hyp_r_pl_emb, self.cur[0])
            res = self.manifold.proj(res, self.cur[0])

            hyp_t_emb = self.manifold.proj_tan0(self.t_emb_list[hop].view(-1, dim), self.cur[0])
            # hrot = self.manifold.sqdist(res, hyp_t_emb.view(-1, dim), self.cur[0])
            hrot = self.dc.forward(self.manifold.sqdist(res, hyp_t_emb.view(-1, dim), self.cur[0]))

            kge_loss += hrot.mean()

        kge_loss = -self.kge_weight * kge_loss

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
            h_emb_sp = self.t_emb_list[hop].shape
            batch_men, men, dim = h_emb_sp[0], h_emb_sp[1], h_emb_sp[2]

            hyp_h_emb = self.manifold.proj_tan0(self.h_emb_list[hop].view(-1, dim), self.cur[0])
            # if hyp_h_emb.isnan().any():
            #     print('hyp_h_emb = ', hyp_h_emb)
            #     input()

            h_emb_rot_q = givens_rotations(self.r_diag_list[hop].view(-1, dim), hyp_h_emb.view(-1, dim)).view((-1, 1, dim))
            h_emb_ref_q = givens_rotations(self.r_diag_2_list[hop].view(-1, dim), hyp_h_emb.view(-1, dim)).view((-1, 1, dim))

            cands = torch.cat([h_emb_rot_q, h_emb_ref_q], dim=1)
            att_weights = torch.sum(self.r_daig_att[hop].view(-1, 1, dim) * cands * self.scale, dim=-1, keepdim=True)
            att_weights = F.softmax(att_weights, dim=1)

            # if att_weights.isnan().any():
            #     print('att_weights = ', att_weights)
            #     input()

            att_q = torch.sum(att_weights * cands, dim=1)
            lhs = self.manifold.proj_tan0_exp(att_q, self.cur[0])

            # if lhs.isnan().any():
            #     print('lhs = ', lhs)
            #     input()

            hyp_r_pl_emb = self.manifold.proj_tan0_exp(self.r_plus_emb_list[hop].view(-1, dim), self.cur[0])
            res = self.manifold.mobius_add(lhs, hyp_r_pl_emb, self.cur[0])
            res = self.manifold.proj(res, self.cur[0])

            hyp_item_emb = self.manifold.proj_tan0_exp(self.item_embeddings, self.cur[0])
            Ri = (hyp_item_emb.unsqueeze(1)).repeat(1,men,1)

            # if res.isnan().any():
            #     print('res = ', res)
            #     input()

            # if Ri.isnan().any():
            #     print('Ri = ', Ri)
            #     input()

            probs = self.manifold.sqdist(res, Ri.view(-1, dim), self.cur[0])
            probs = torch.exp(- self.att_beta * probs - 0.05)

            # if probs.isnan().any():
            #     print('probs = ', probs)
            #     input()

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
        kle_o_list = torch.cat(kle_o_list, dim=1)
        kle_shape = kle_o_list.shape
        probs = self.one_like_bahop[:kle_shape[0],:]
        probs = self.manifold.lorentz_factor(kle_o_list.view(-1,kle_shape[-1]), c=self.cur[0]) * probs.view(-1,1)
        probs = probs.view(kle_shape[0], kle_shape[1])
        probs_expanded = (probs/(probs.sum(1).unsqueeze(1))).unsqueeze(-1)
        kle_o = (kle_o_list * probs_expanded).sum(dim=1)
        hyp_o = self.manifold.klein_to(kle_o, self.cur[0])
        hyp_o = self.manifold.proj(hyp_o, self.cur[0])

        dim = item_embeddings.shape[-1]

        hyp_item_emb = self.manifold.proj_tan0(item_embeddings.view(-1, dim), self.cur[0])

        # if hyp_item_emb.isnan().any():
        #     print('hyp_item_emb 2 = ', hyp_item_emb)
        #     input()


        it_emb_rot_q = givens_rotations(self.pre_diag_emb.view(-1, dim), hyp_item_emb.view(-1, dim)).view((-1, 1, dim))
        it_emb_ref_q = givens_rotations(self.pre_diag_2_emb.view(-1, dim), hyp_item_emb.view(-1, dim)).view((-1, 1, dim))

        cands = torch.cat([it_emb_rot_q, it_emb_ref_q], dim=1)
        att_weights = torch.sum(self.pre_daig_att.view(-1, 1, dim) * cands * self.scale, dim=-1, keepdim=True)
        att_weights = F.softmax(att_weights, dim=1)

        att_q = torch.sum(att_weights * cands, dim=1)

        # if att_q.isnan().any():
            # print('att_q 2 = ', att_q)
            # input()

        lhs = self.manifold.proj_tan0_exp(att_q, self.cur[0])

        # if lhs.isnan().any():
        #     print('lhs 2 = ', lhs)
        #     input()

        hyp_r_pl_emb = self.manifold.proj_tan0_exp(self.pre_plus_emb.view(-1, dim), self.cur[0])

        # if hyp_r_pl_emb.isnan().any():
        #     print('hyp_r_pl_emb 2 = ', hyp_r_pl_emb)
        #     input()

        item_rot = self.manifold.mobius_add(lhs, hyp_r_pl_emb, self.cur[0])
        item_rot = self.manifold.proj(item_rot, self.cur[0])

        return self.dc.forward(self.manifold.sqdist(hyp_o, item_rot, self.cur[0]))

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
        # scores = return_dict["scores"].detach().cpu().numpy()
        # labels = labels.cpu().numpy()
        scores = return_dict["scores"].cpu().detach().numpy() 
        labels = labels.cpu().detach().numpy() 
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc, f1


    def get_scores(self, user, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(user, items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].cpu().detach().numpy()
        return items.tolist(), scores



    def tsne_embedding(self, args, epoch):
        embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(self.entity_emb_matrix_sw.weight.cpu().detach().numpy()[list(args.entities_set.keys()),:])
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        args.path.open_pic_dir()

        pd_x, pd_y, pd_z, pd_color = [], [], [], []
        color_dict = {}

        for item, value in args.entities_type_set.items():
            index = list(args.entities_type_set[item].keys())
            enti_num = len(list(args.entities_type_set[item].keys()))
            if len(index) < 30:
                continue
            vis_x = embeddings[index, 0]
            vis_y = embeddings[index, 1]

            pd_x.extend(vis_x.tolist())
            pd_y.extend(vis_y.tolist())
            pd_z.extend([item] * enti_num)
            pd_color.extend(args.entities_type_color[item])

            color_dict[item] = args.entities_type_color[item]

        data_dic = pd.DataFrame({'x': pd_x, 'y': pd_y, 'z': pd_z})

        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("ticks")
        sns.lmplot(x = 'x', y = 'y', data = data_dic, fit_reg=False, legend=True, palette=color_dict , hue='z', size=9, scatter_kws={'s': 30,'alpha':0.3})
        plt.tick_params(labelsize=10)
        plt.savefig(args.path.pic_file + '/TSNE_type_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
        plt.close()

    # def tsne_embedding(self, args, epoch):
    #     # embeddings = TSNE(n_jobs=4).fit_transform(self.entity_emb_matrix.weight)
    #     # self.manifold.proj_tan0_exp(att_q, self.cur[0])
    #     embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(self.entity_emb_matrix_sw.weight.cpu().detach().numpy()[list(args.entities_set.keys()),:])
    #     vis_x = embeddings[:, 0]
    #     vis_y = embeddings[:, 1]
    #     plt.scatter(vis_x, vis_y, s=args.plt_s,  marker='.')
    #     args.path.open_pic_dir()
    #     plt.savefig(args.path.pic_file + '/TSNE_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
    #     plt.close()

    #     for item, value in args.entities_type_set.items():
    #         index = list(args.entities_type_set[item].keys())
    #         if len(index) < 30:
    #             continue
    #         # print('item color = ',item,  args.entities_type_color[item])

    #         # print('index = ', index)
    #         # input()

    #             # embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(self.entity_emb_matrix_sw.weight.cpu().detach().numpy()[list(args.entities_type_set[item].keys()),:])
    #         vis_x = embeddings[index, 0]
    #         vis_y = embeddings[index, 1]


    #         # print('vis_x, vis_y = ', vis_x, vis_y)
    #         # input()

    #         # plt.scatter(vis_x, vis_y, c=args.entities_type_color[item], s=args.plt_s, label=item, alpha=0.3, edgecolors='none')
    #         plt.scatter(vis_x, vis_y, c=args.entities_type_color[item], s=args.plt_s, label=item)
            

    #     # legend1 = plt.legend(*scatter.legend_elements(num=len(args.entities_type_set)),
    #     #                     loc="upper left", title="labels")
    #     # plt.add_artist(legend1)
    #     plt.legend()
    #     # ax.grid(True)

    #     plt.savefig(args.path.pic_file + '/TSNE_type_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
    #     plt.close()

# args.entities_type_set

