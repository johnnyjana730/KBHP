import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module


from sklearn.metrics import f1_score, roc_auc_score
import model.manifolds as manifolds

from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from tsnecuda import TSNE
import seaborn as sns
import pandas as pd  


class FermiDiracDecoder(Module):
    """Fermi Dirac to compute edge probabilities based on distances."""

    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

class EpNet_base(nn.Module):
    def __init__(self, args, n_entity, n_relation):
        super(EpNet_base, self).__init__()
        self._parse_args(args, n_entity, n_relation)
        # self._build_embeddings()        

        self.entity_emb_matrix = nn.Embedding(self.n_entity, self.dim)
        self.entity_emb_matrix.weight.data =  torch.randn((self.n_entity, self.dim))

        self.rela_plus_emb_matrix = nn.Embedding(self.n_relation, self.dim)
        self.rela_plus_emb_matrix.weight.data = torch.randn((self.n_relation, self.dim))


        self.relation_emb_matrix = nn.Embedding(self.n_relation, self.dim * self.dim)
        self.relation_emb_matrix.weight.data = torch.randn(self.n_relation, self.dim * self.dim)

        self.transform_matrix = nn.Linear(self.dim, self.dim, bias=True)

        self.criterion = nn.BCELoss()
        self.dropout = 0.5

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
    
        self.uemb_generate = self._key_addressing
        self.fin_predict = self.predict

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
        
        # print('scores = ', scores)
        # print('labels = ', labels)
        base_loss = self.criterion(scores, labels.float())
        loss = base_loss
        return dict(base_loss=base_loss, kge_loss=0, l2_loss=0, loss=loss)

    def proj_tan0_exp(self, input_ecu, cur):
        input_tan = self.manifold.proj_tan0(input_ecu, cur) 
        output_hyp = self.manifold.expmap0(input_tan, cur)
        output_hyp = self.manifold.proj(output_hyp, cur)
        return output_hyp

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

            # self.item_embeddings = self._update_item_embedding(item_embeddings, o)

            o_list.append(o)

        return o_list

    def predict(self, item_embeddings, o_list):
        y = o_list[-1]
        if self.using_all_hops:
            for i in range(self.n_hop - 1):
                y += o_list[i]
        # [batch_size]
        scores = (item_embeddings * y).sum(dim=1)
        return torch.sigmoid(scores)

    def evaluate(self, user, items, labels, memories_h, memories_r, memories_t):
        return_dict = self.forward(user, items, labels, memories_h, memories_r, memories_t)
        labels = labels.cpu().detach().numpy() 

        scores = return_dict["scores"].cpu().detach().numpy()
        auc = roc_auc_score(y_true=labels, y_score=scores)
        predictions = [1 if i >= 0.5 else 0 for i in scores]
        f1 = f1_score(y_true=labels, y_pred=predictions)
        acc = np.mean(np.equal(predictions, labels))
        return auc, acc, f1


    def tsne_embedding(self, args, epoch):
        # embeddings = TSNE(n_jobs=4).fit_transform(self.entity_emb_matrix.weight)
        # self.manifold.proj_tan0_exp(att_q, self.cur[0])
        embeddings = TSNE(n_components=2, perplexity=15, learning_rate=10).fit_transform(self.entity_emb_matrix_sw.weight.cpu().detach().numpy()[list(args.entities_set.keys()),:])
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, s=args.plt_s,  marker='.')
        args.path.open_pic_dir()
        plt.savefig(args.path.pic_file + '/TSNE_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
        plt.close()

        for item, value in args.entities_type_set.items():
            index = list(args.entities_type_set[item].keys())
            if len(index) < 30:
                continue
            vis_x = embeddings[index, 0]
            vis_y = embeddings[index, 1]
            plt.scatter(vis_x, vis_y, c=args.entities_type_color[item], s=args.plt_s, label=item)
        plt.legend()
        plt.savefig(args.path.pic_file + '/TSNE_type_sw{:d}_fig{:d}.png'.format(args.sw_stage, epoch))
        plt.close()
