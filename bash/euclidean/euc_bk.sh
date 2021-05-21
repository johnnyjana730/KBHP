export CUDA_VISIBLE_DEVICES=$1

# elif [ $3 = "bk" ]
# then
#     DATASET=('Book-Crossing')
#     lr=1.5e-3
#     batch_size=1024

dataset="Book-Crossing"
# model hyper-parameter setting
dim=8
n_hop=2

n_memory=16
l2_weight=1e-5
kge_weight=1e-2

tolerance=5
early_stop=5
batch_size=256

lr=1e-3
manifold_name=Euclidean
model_name=EpNet_item_tr

NEIGHBOR_SIZE=(64 32 16)
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
