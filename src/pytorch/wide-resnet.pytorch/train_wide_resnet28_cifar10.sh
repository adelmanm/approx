#!/bin/bash
# example usage: ./train_wide_resnet28_cifar100.sh opt_norep_noscale 0.5 10 gpu0 fwd 
algorithm=$1
sample_ratio=$2
minimal_k=$3
lr=0.15
gpuid=$4
prefix=$5
suffix=$6

config_str=wrn28-10_${algorithm}_${prefix}_${sample_ratio}${suffix}_minimal_k_${minimal_k}
log_dir=${gpuid}_logs
output_log=${log_dir}/${config_str}.log
run_cmd_train="stdbuf -oL -eL python3 main.py --lr ${lr} --net_type 'wide-resnet' --depth 28 --widen_factor 10 --dropout 0 --dataset 'cifar10' --batch_size 256 --sample_ratio ${sample_ratio} --minimal_k ${minimal_k} &>> ${output_log}"


echo "running wide resnet 28 training script" | tee ${output_log}
echo "algorithm is $algorithm $prefix $suffix" | tee -a ${output_log}
echo "sample_ratio is $sample_ratio" | tee -a ${output_log}
echo "minimal_k is $minimal_k" | tee -a ${output_log}
echo "lr is $lr" | tee -a ${output_log}

echo "${run_cmd_train}" | tee -a ${output_log}
eval $run_cmd_train

