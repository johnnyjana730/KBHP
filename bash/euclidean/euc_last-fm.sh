export CUDA_VISIBLE_DEVICES=1

dataset="last-fm_50core"
# model hyper-parameter setting
dim=16
n_hop=$2

n_memory=64
l2_weight=1e-7

# learning parameter setting
batch_size=512
# tolerance=2
# early_stop=5

lr=1e-2
manifold_name=Euclidean
model_name=EpNet_item_tr

NEIGHBOR_SIZE=(64 32 16)
HOP_SIZE=(1 2)

for NEIGHBOR in ${NEIGHBOR_SIZE[@]}
do
    for hop_sz in ${HOP_SIZE[@]}
    do

        n_memory=$NEIGHBOR
        cmd="python3 ../src/main_get.py
            --dataset $dataset
            --dim $dim
            --manifold_name $manifold_name
            --model_name $model_name
            --n_memory $n_memory
            --l2_weight $l2_weight
            --batch_size $batch_size
            --lr $lr
            --n_hop $hop_sz"
        $cmd
    done
done
