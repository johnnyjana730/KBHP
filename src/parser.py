import argparse
from path_svemb import Path

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='movie', help='which dataset to use')
	parser.add_argument('--dim', type=int, default=16, help='dimension of entity and relation embeddings')
	parser.add_argument('--n_hop', type=int, default=2, help='maximum hops')
	parser.add_argument('--kge_weight', type=float, default=0.0005, help='weight of the KGE term')
	parser.add_argument('--l2_weight', type=float, default=1e-7, help='weight of the l2 regularization term')
	parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
	parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
	parser.add_argument('--n_epoch', type=int, default=1000, help='the number of epochs')
	parser.add_argument('--n_memory', type=int, default=32, help='size of ripple set for each hop')
	parser.add_argument('--emb_eval', type=int, default=1, help='size of ripple set for each hop')

	parser.add_argument('--r', type=float, default=2.0, help='the number of epochs')
	parser.add_argument('--plt_s', type=float, default=0.2, help='the number of epochs')
	parser.add_argument('--dis_mx_clip', type=float, default=15.0, help='the number of epochs')
	parser.add_argument('--t', type=float, default=1., help='size of ripple set for each hop')
	parser.add_argument('--et_freq', type=float, default=3, help='size of ripple set for each hop')


	parser.add_argument('--item_update_mode', type=str, default='plus_transform', help='how to update item at the end of each hop')
	parser.add_argument('--save_model_name', type=str, default='tmp', help='how to update item at the end of each hop')
	parser.add_argument('--expname', type=str, default='', help='how to update item at the end of each hop')

	parser.add_argument('--model_name', type=str, default='plus_transform', help='model_name')
	parser.add_argument('--using_all_hops', type=bool, default=True, help='whether using outputs of all hops or just the last hop when making prediction')
	parser.add_argument('--show_topk', type=int, default=0, help='whether')
	parser.add_argument('--save_record_user_list', type=bool, default=False, help='number of iterations when computing entity representation')

	parser.add_argument('--manifold_name', type=str, default='Hyperboloid', help='[Hyperboloid, PoincareBall, Euclidean]')
	parser.add_argument('--att_beta', type=float, default=0.05, help='weight of the KGE term')
	parser.add_argument('--use_cuda', type=int, default=1, help='whether to use gpu')
	parser.add_argument('--grad_clip_max', type=float, default=1e-05, help='weight of the KGE term')
	parser.add_argument('--grad_clip', type=int, default=1, help='weight of the KGE term')

	parser.add_argument('--sw_stage', type=int, default=0, help='weight of the KGE term')

	parser.add_argument('--lr_reduce_freq', type=int, default=10, help='weight of the KGE term')
	parser.add_argument('--gamma', type=float, default=0.8, help='weight of the KGE term')
	parser.add_argument('--debug_test', type=int, default=0, help='weight of the KGE term')

	parser.add_argument('--load_pretrain_emb', type=bool, default=False, help='load pretrain emb')

	args = parser.parse_args()
	path = Path(args)
	args.path = path

	if args.dataset == 'MovieLens-1M':
		args.et_freq = 0
	elif args.dataset == 'last-fm_50core':
		args.et_freq = 10
	elif args.dataset == 'amazon-book_20core':
		args.et_freq = 5
	elif args.dataset == 'music':
		args.et_freq = 0

	args.debug_test = (args.debug_test == 1)	
	args.use_cuda = (args.use_cuda == 1)
	args.grad_clip = (args.grad_clip == 1)
	args.show_topk = (args.show_topk == 1)
	args.emb_eval = (args.emb_eval == 1)

	return args