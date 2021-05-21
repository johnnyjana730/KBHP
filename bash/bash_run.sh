export PYTHONPATH="./"

gpu=$1
dataset=$2

bash hyperbolic/run_bash.sh $gpu $dataset &
