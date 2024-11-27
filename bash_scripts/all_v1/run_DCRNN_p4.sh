# 命令行传入 seed
seed=${1:-0}
task_name="Main-${seed}"
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 模型配置
ModelList=("DCRNN")

# 运行
HisLenList=(256)
PredLenList=(6 12 48)
batch_size=0  # 设置 0 开启 AutoBatch
# 共 16 个数据集
# DatasetList=('crypto12' 'COVID_CHI' 'COVID_DEATHS' 'stocknet' 'COVID_US' 'METRO_HZ' 'ETT_hour' 'JONAS_NYC_bike' 'weather' 'JONAS_NYC_taxi' 'Jena_climate' 'METRO_SH' 'nasdaq100' 'Metr-LA' 'electricity' 'PEMSBAY')
DatasetList=('crypto12' 'stocknet' 'METRO_HZ' 'ETT_hour' 'JONAS_NYC_bike' 'weather' 'JONAS_NYC_taxi' 'Jena_climate' 'METRO_SH' 'nasdaq100' 'Metr-LA' 'electricity' 'PEMSBAY')
# 时间统计
# +----------------+----------+
# |  dataset_name  | one_epoch|
# +----------------+----------+
# |    crypto12    |   1.77   |
# |   COVID_CHI    |   6.43   | *
# |  COVID_DEATHS  |  6.958   | *
# |    stocknet    |  13.123  |
# |    COVID_US    |  15.434  | *
# |    METRO_HZ    |  19.256  |
# |    ETT_hour    |  29.172  |
# | JONAS_NYC_bike |  92.549  |
# |    weather     | 107.014  |
# | JONAS_NYC_taxi | 137.853  |
# |  Jena_climate  | 703.342  |
# |    METRO_SH    | 1017.638 |
# |   nasdaq100    | 1253.488 |
# |    Metr-LA     | 2163.695 |
# |  electricity   | 3774.79  |
# |     PEMSBAY    |  unkown  |
# |     PEMS03     | 4553.315 | x
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
                    --dataset $dataset --model $model --debug
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