# 命令行传入 seed
seed=${1:-0}
task_name="SearchInput-${seed}"
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 模型配置
# ModelList=("AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" )
ModelList=("Mamba" "MTGNN" "ST_Norm" "TTS_Norm" "TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "STID" "DCRNN" "STWA" "NET3")


# ==================== Nature Datasets ====================
HisLenList=(6 12 24)
PredLenList=(6 12 24)
batch_size=1  # 设置 1
DatasetList=('COVID_DEATHS')

for pred_len in ${PredLenList[@]}; do
    for his_len in ${HisLenList[@]}; do
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


# 运行
# HisLenList=(12 48 96)
# # PredLenList=(6 12 48)
# PredLenList=(12 48 96)
# batch_size=0  # 设置 0 开启 AutoBatch
# # 共 16 个数据集
# # DatasetList=('COVID_DEATHS' 'crypto12' 'METRO_HZ' 'COVID_CHI' 'stocknet' 'ETT_hour' 'COVID_US' 'JONAS_NYC_taxi' 'JONAS_NYC_bike' 'weather' 'Metr-LA' 'electricity' 'METRO_SH' 'Jena_climate' 'nasdaq100' 'PEMSBAY')
# # DatasetList=('crypto12' 'METRO_HZ' 'stocknet' 'ETT_hour' 'JONAS_NYC_taxi' 'JONAS_NYC_bike' 'weather' 'Metr-LA' 'electricity' 'METRO_SH' 'Jena_climate' 'nasdaq100')
# DatasetList=('COVID_CHI')

# # ==================== Traffic + Weather + Energy + Finance Datasets ====================
# for dataset in ${DatasetList[@]}; do
#     for his_len in ${HisLenList[@]}; do
#         for pred_len in ${PredLenList[@]}; do
#             for model in ${ModelList[@]}; do
#                 # -------------------------------------------
#                 python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
#                     --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
#                     --normalizer $normalizer --graph_init $graph_init \
#                     --dataset_base $dataset_base \
#                     --scheduler $scheduler \
#                     --lr_finder \
#                     --logger $logger \
#                     --seed $seed \
#                     --dataset $dataset --model $model --debug
#             done
#         done
#     done
# done


