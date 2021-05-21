from parser import parse_args

from model.hyperplan.hyper_rela_bi_pur import hpNet_rela_bi_pur
from model.hyperplan.hyper_rela_rot_pur_ih import hpNet_rela_rot_pur_ih
from model.hyperplan.hyper_rela_rot_pur_hih import hpNet_rela_rot_pur_hih

args = parse_args()

if args.model_name == 'bi_pur':
    Model = hpNet_rela_bi_pur

elif args.model_name == 'rot_pur_ih':
    Model = hpNet_rela_rot_pur_ih

elif args.model_name == 'rot_pur_hih':
    Model = hpNet_rela_rot_pur_hih