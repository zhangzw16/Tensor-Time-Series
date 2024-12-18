# 命令行传入 seed
seed=${1:-0}
task_name="Main-${seed}"
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 模型配置
ModelList=("NET3")

# 运行
HisLenList=(96)
PredLenList=(12 48)
batch_size=0  # 设置 0 开启 AutoBatch
DatasetList=('crypto12' 'METRO_HZ' 'stocknet' 'ETT_hour' 'JONAS_NYC_taxi' 'JONAS_NYC_bike' 'weather' 'Metr-LA' 'electricity' 'METRO_SH' 'Jena_climate' 'nasdaq100')
# 时间统计
# +----------------+----------+
# |  dataset_name  | one_epoch|
# +----------------+----------+
# |  COVID_DEATHS  |  2.391   | *
# |    crypto12    |   2.95   |
# |    METRO_HZ    |  10.188  |
# |   COVID_CHI    |  13.702  | *
# |    stocknet    |  14.834  |
# |    ETT_hour    |  23.953  |
# |    COVID_US    |  34.166  | *
# | JONAS_NYC_taxi |  39.531  |
# | JONAS_NYC_bike |  41.327  |
# |    weather     |  97.761  |
# |    Metr-LA     | 216.582  |
# |     PEMS03     | 240.162  | x
# |  electricity   | 257.493  |
# |    METRO_SH    | 507.109  |
# |  Jena_climate  | 627.238  |
# |   nasdaq100    | 855.812  |
# |     PEMSBAY    |  unkown  |
# |     PEMS07     | 1995.478 | x
# +----------------+----------+

for his_len in ${HisLenList[@]}; do
    for pred_len in ${PredLenList[@]}; do
        for model in ${ModelList[@]}; do
            for dataset in ${DatasetList[@]}; do
                # -------------------------------------------
                python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --dataset_base $dataset_base \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --seed $seed \
                    --dataset $dataset --model $model
            done
        done
    done
done


# ==================== Nature Datasets ====================
# HisLenList=(6 12 24)
# PredLenList=(1 6 12)
# batch_size=1  # 设置 1
# DatasetList=('COVID_DEATHS' 'COVID_CHI' 'COVID_US')
# for his_len in ${HisLenList[@]}; do
#     for pred_len in ${PredLenList[@]}; do
#         for model in ${ModelList[@]}; do
#             for dataset in ${DatasetList[@]}; do
#                 # -------------------------------------------
#                 python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
#                     --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
#                     --normalizer $normalizer --graph_init $graph_init \
#                     --dataset_base $dataset_base \
#                     --scheduler $scheduler \
#                     --lr_finder \
#                     --logger $logger \
#                     --seed $seed \
#                     --dataset $dataset --model $model
#             done
#         done
#     done
# done