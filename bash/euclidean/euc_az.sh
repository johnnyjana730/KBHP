export CUDA_VISIBLE_DEVICES=$1

dataset="amazon-book_20core"
# model hyper-parameter setting
dim=16
n_hop=$2

n_memory=32
l2_weight=1e-6

tolerance=5
early_stop=5
batch_size=1024
kge_weight=0.05

lr=5e-3
manifold_name=Euclidean
model_name=EpNet_item_tr

NEIGHBOR_SIZE=(16 32 64)
HOP_SIZE=(2 1)

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
            --kge_weight $kge_weight
            --n_memory $n_memory
            --l2_weight $l2_weight
            --batch_size $batch_size
            --lr $lr
            --n_hop $hop_sz"
        $cmd
    done
done
