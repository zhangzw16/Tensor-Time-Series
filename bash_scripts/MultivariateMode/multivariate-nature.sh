# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="MultiVariate-${seed}"

# 模型配置
ModelList=('TimesNet' 'StemGNN' 'AutoFormer' 'CrossFormer' 'PatchTST' 'DLinear' 'NLinear' 'STID' 'STWA')
# ModelList=("Mamba" "MTGNN" "ST_Norm" "TTS_Norm" "TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "STID")

# ==================== Nature Datasets ====================
HisLenList=()
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