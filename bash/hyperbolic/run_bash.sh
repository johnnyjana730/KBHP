export CUDA_VISIBLE_DEVICES=$1

if [ $2 = "mv" ]
then
    dataset='MovieLens-1M'
    lr=6e-4
    batch_size=512
	kge_weight=1e-6
	l2_weight=1e-7
    beta_sz=0.05
    n_memory=96
    r=5.0
	t=1.0
	dim=16
	hop_sz=1
	model_name=rot_pur_hih
elif [ $2 = "az" ]
then
    dataset='amazon-book_20core'
    lr=5e-4
    batch_size=512
	kge_weight=5e-5
	l2_weight=5e-7
    beta_sz=0.15
	n_memory=16
	r=2.0
	t=1.0
	dim=16
	hop_sz=2
	model_name=rot_pur_hih
elif [ $2 = "la" ]
then
    dataset="last-fm_50core"
    lr=1.0e-3
    batch_size=512
	kge_weight=1e-7
	l2_weight=1e-7
    beta_sz=0.1
	n_memory=64
	r=5.0
	t=1.0
	dim=32
	hop_sz=1
	model_name=rot_pur_hih
elif [ $2 = "mu" ]
then
    dataset='music'
    lr=8e-4
    batch_size=128
	kge_weight=1e-5
	l2_weight=1e-7
	n_memory=64
    beta_sz=0.05
	r=2.0
	t=1.0
	dim=32
	hop_sz=1
	model_name=rot_pur_ih
fi

expname='KBHP'
emb_eval=0
debug_test=0
use_cuda=1
maniforld_na=Hyperboloid

save_record_user_list=1
save_model_name=topk

cmd="python3 ../src/main.py --dataset $dataset --dim $dim  --att_beta $beta_sz --n_memory $n_memory \
     --manifold_name $maniforld_na --model_name $model_name --l2_weight $l2_weight --debug_test $debug_test \
     --kge_weight $kge_weight --use_cuda $use_cuda --emb_eval $emb_eval --r ${r} --t ${t} --batch_size $batch_size \
     --lr $lr --n_hop $hop_sz --expname ${expname} --save_model_name $save_model_name"
$cmd
