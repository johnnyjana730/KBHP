import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from sklearn.metrics import f1_score, roc_auc_score
from model.euclidean.Eucl_base_ini import *
import model.manifolds as manifolds

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from tsnecuda import TSNE
import seaborn as sns
import pandas as pd  

class EpNet_item_tr_ini(Eucl_base_ini):
    def __init__(self, args, n_entity, n_relation):
        super().__init__(args, n_entity, n_relation)


    def get_scores(self, user, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(user, items, labels, memories_h, memories_r, memories_t)
        scores = return_dict["scores"].cpu().detach().numpy()
        return items.tolist(), scores

    # def forward(self, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):
    def forward(self, users: torch.LongTensor, items: torch.LongTensor, labels: torch.LongTensor, memories_h: list, memories_r: list, memories_t: list):
        # [batch size, dim]
        self.item_embeddings = self.entity_emb_matrix(items)

        self.h_emb_list = []
        self.t_emb_list = []
        self.r_emb_list = []
        self.r_plus_emb_list = []
        self.cur = []
        for i in range(self.n_hop):
            h_emb = self.entity_emb_matrix(memories_h[i])
            t_emb = self.entity_emb_matrix(memories_t[i])
            r_plus_emb = self.rela_plus_emb_matrix(memories_r[i])

            self.cur.append(1.0)
            r_emb = self.relation_emb_matrix(memories_r[i]).view(-1, self.n_memory, self.dim, self.dim)
            # [batch size, n_memory, dim]
            self.h_emb_list.append(h_emb)
            # [batch size, n_memory, dim]
            self.t_emb_list.append(t_emb)
            # [batch size, n_memory, dim, dim]
            self.r_emb_list.append(r_emb)
            # [batch size, n_memory, dim]
            self.r_plus_emb_list.append(r_plus_emb)
            # self.uemb_generate = self._key_hper_addressing
            # self.fin_predict = self.hyper_predict

        o_list = self.uemb_generate()
        scores = self.fin_predict(self.item_embeddings, o_list).squeeze()

        return_dict = self._compute_loss(scores, labels)
        return_dict["scores"] = scores

        return return_dict

    def _compute_loss(self, scores, labels):
        base_loss = self.criterion(scores, labels.float())

        kge_loss = 0
        for hop in range(self.n_hop):
            # [batch size, n_memory, 1, dim]
            h_expanded = torch.unsqueeze(self.h_emb_list[hop], dim=2)
            # [batch size, n_memory, dim, 1]
            t_expanded = torch.unsqueeze(self.t_emb_list[hop], dim=3)
            # [batch size, n_memory, dim, dim]
            hRt = torch.squeeze(
                torch.matmul(torch.matmul(h_expanded, self.r_emb_list[hop]), t_expanded)
            )
            kge_loss += torch.sigmoid(hRt).mean()
        kge_loss = -self.kge_weight * kge_loss

        l2_loss = 0
        for hop in range(self.n_hop):
            l2_loss += (self.h_emb_list[hop] * self.h_emb_list[hop]).sum()
            l2_loss += (self.t_emb_list[hop] * self.t_emb_list[hop]).sum()
            l2_loss += (self.r_emb_list[hop] * self.r_emb_list[hop]).sum()
        l2_loss = self.l2_weight * l2_loss

        loss = base_loss + kge_loss + l2_loss
        return dict(base_loss=base_loss, kge_loss=0, l2_loss=0, loss=loss)


    def _key_addressing(self):
        o_list = []
        for hop in range(self.n_hop):
            # [batch_size, n_memory, dim, 1]
            h_expanded = torch.unsqueeze(self.h_emb_list[hop], dim=3)

            # [batch_size, n_memory, dim]
            Rh = torch.squeeze(torch.matmul(self.r_emb_list[hop], h_expanded))

            # [batch_size, dim, 1]
            v = torch.unsqueeze(self.item_embeddings, dim=2)

            # [batch_size, n_memory]
            probs = torch.squeeze(torch.matmul(Rh, v))

            # [batch_size, n_memory]
            probs_normalized = F.softmax(probs, dim=1)

            # [batch_size, n_memory, 1]
            probs_expanded = torch.unsqueeze(probs_normalized, dim=2)

            # [batch_size, dim]
            o = (self.t_emb_list[hop] * probs_expanded).sum(dim=1)

            self.item_embeddings = self._update_item_embedding(self.item_embeddings, o)

            o_list.append(o)

        return o_list

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        for i in range(self.n_hop - 1):
            y += o_list[i]
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)

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


    def tsne_embedding(self, args, epoch):

        # embeddings = TSNE(n_jobs=4).fit_transform(self.entity_emb_matrix.weight)
        # self.manifold.proj_tan0_exp(att_q, self.cur[0])
        embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(self.entity_emb_matrix.weight.cpu().detach().numpy()[list(args.entities_set.keys()),:])
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        # plt.scatter(vis_x, vis_y, s=args.plt_s,  marker='.')
        args.path.open_pic_dir()
        # plt.savefig(args.path.pic_file + '/TSNE_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
        # plt.close()

        # data_dic = pd.DataFrame(columns=['x','y','z', 'color'])
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

            # print('vis_x.tolist() = ', len(vis_x.tolist()))
            # print('vis_y.tolist() = ', len(vis_y.tolist()))
            # print('[item] * enti_num= ', len([item] * enti_num))
            # print('[args.entities_type_color[item]] * enti_num = ', len([args.entities_type_color[item]] * enti_num))

            # tmp = pd.DataFrame({'x': vis_x.tolist(), 'y': vis_y.tolist(), 'z': [item] * enti_num, 'color': [args.entities_type_color[item]] * enti_num})
            # tmp = pd.DataFrame(zip([vis_x.tolist(), vis_y.tolist(),[item] * enti_num, [args.entities_type_color[item]] * enti_num]), columns=['x','y','z', 'color'])
            # data_dic = data_dic.append(tmp, ignore_index=True)

            # print('data_dic = ', data_dic)
            # input()
            # a  =pd.DataFrame({'x':a[0], 'y':a[1] , 'z':a[2]})

            # plt.scatter(vis_x, vis_y, c=args.entities_type_color[item], s=args.plt_s, label=item)
        # pd.DataFrame(zip([vis_x.tolist(), vis_y.tolist(),[item] * enti_num, [args.entities_type_color[item]] * enti_num]), columns=['x','y','z', 'color'])

        # print('pd_x= ', len(pd_x))
        # print('pd_y = ', len(pd_y))
        # print('pd_z = ', len(pd_z))
        # print('pd_color = ', len(pd_color))

        data_dic = pd.DataFrame({'x': pd_x, 'y': pd_y, 'z': pd_z})

        sns.set_context("notebook", font_scale=1.1)
        sns.set_style("ticks")
        # sns.lmplot(x = 'x', y = 'y', data = data_dic, fit_reg=False, legend=True, size = 0.5, hue='z', scatter_kws={'facecolors': 'color', 'alpha':0.3})
        sns.lmplot(x = 'x', y = 'y', data = data_dic, fit_reg=False, legend=True, palette=color_dict , hue='z', size=9, scatter_kws={'s': 30,'alpha':0.3})
        # plt.title('t-SNE ' + str(args.model_name)).set_fontsize('8')
        # plt.xlabel("dim 1 ").set_fontsize('8')
        # plt.ylabel("dim 2 ").set_fontsize('8')
        plt.tick_params(labelsize=16)
        # plt.title('t-SNE ' + str(args.model_name), weight='bold').set_fontsize('8')
        # plt.xlabel("dim 1 ", weight='bold').set_fontsize('8')
        # plt.ylabel("dim 2 ", weight='bold').set_fontsize('8')
        plt.savefig(args.path.pic_file + '/TSNE_type_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
        plt.close()
