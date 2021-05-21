import os
from datetime import datetime

now = datetime.now() 

date_time = now.strftime("%m_%d")
print("date and time:",date_time)   

class Path:
    def __init__(self, args):
        self.data = f'../data/{args.dataset}/'
        self.misc = f'../misc/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
        self.emb = f'../misc/{args.dataset}/emb/{str(args.model_name)}/{str(args.manifold_name)}/'

        self.pre_emb = self.emb + '_sw_' + str(args.sw_stage - 1) + '.ckpt'

        if args.show_topk == 1:
            self.log_file = f'../output/log_tpk/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
            self.eva_file = f'../output/eva_tpk/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
            self.pic_file = f'../output/pic_tpk/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
        else:
            self.log_file = f'../output/log/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
            self.eva_file = f'../output/eva/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'
            self.pic_file = f'../output/pic/{args.dataset}/{str(args.model_name)}/{str(args.manifold_name)}/'

        self.exp_name = str(date_time) + "_" + str(args.expname) + '_hid_' + str(args.dim) + '_hop_' + str(args.n_hop) +  '_mm_' + str(args.n_memory)  + \
                        '_lr_' + str(args.lr) + '_ke_' + str(args.kge_weight) + '_l2_' + str(args.l2_weight) + \
                        '_att_beta_' + str(args.att_beta) + '_r_' + str(args.r) + '_t_' + str(args.t)

        self.ensureDir(self.data)
        self.ensureDir(self.misc)
        self.ensureDir(self.emb)
        self.ensureDir(self.log_file)
        self.ensureDir(self.eva_file)

        self.emb += self.exp_name
        self.log_file += self.exp_name
        self.eva_file += self.exp_name
        self.pic_file += self.exp_name + '/'

    def ensureDir(self, dir_path):
        try:
            d = os.path.dirname(dir_path)
            # print('d = ', d)
            if not os.path.exists(d):
                os.makedirs(d)
        except:
            pass

    def open_pic_dir(self):
        self.ensureDir(self.pic_file)