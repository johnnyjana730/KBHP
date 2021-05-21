export CUDA_VISIBLE_DEVICES=$1


l2_weight=5e-5
kge_weight=1e-5

if [ $2 = "mv" ]
then
    dataset=('MovieLens-1M')
    lr=5e-3
    l2_weight=1e-7
    batch_size=512
elif [ $2 = "az" ]
then
    dataset=('amazon-book_20core')
    lr=3e-3
    batch_size=1024
    l2_weight=1e-7
    kge_weight=1e-6

elif [ $2 = "bk" ]
then
    dataset=('Book-Crossing')
    lr=0.6e-3
    l2_weight=5e-5
    kge_weight=1e-6
    batch_size=256
elif [ $2 = "mu" ]
then
    dataset=('music')
    lr=0.6e-3
    batch_size=128
    l2_weight=1e-6
    kge_weight=1e-5
elif [ $2 = "yp" ]
then
    dataset=('yelp2018_20core')
    lr=1e-2
    batch_size=1024
elif [ $2 = "la" ]
then
    dataset=("last-fm_50core")
    lr=1e-3
    batch_size=512
fi

dim=16
n_memory=32
tolerance=5
early_stop=5

# l2_weight=5e-5
# kge_weight=1e-5
# KGE_weight=(1e-5)
# L2_weight=(5e-7)
emb_eval=0
manifold_name=Euclidean
model_name=EpNet_item_tr_ini
expname='emb_vis_fl_frame'

# NEIGHBOR_SIZE=(16 32 64 8)
DIM_SIZE=(64 32 8 4)
NEIGHBOR_SIZE=(16)
HOP_SIZE=(1)

for NEIGHBOR in ${NEIGHBOR_SIZE[@]}
do
    for hop_sz in ${HOP_SIZE[@]}
    do

        for dim in ${DIM_SIZE[@]}
        do
            n_memory=$NEIGHBOR
            cmd="python3 ../src/main_get.py
                --dataset $dataset
                --expname $expname
                --dim $dim
                --emb_eval 1
                --manifold_name $manifold_name
                --model_name $model_name
                --kge_weight $kge_weight
                --n_memory $n_memory
                --l2_weight $l2_weight
                --batch_size $batch_size
                --lr $lr
                --n_hop $hop_sz"
            $cmd
        done
    done
done
