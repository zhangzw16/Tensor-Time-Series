# !!!!!!!!!!!!! 需要设置下面三个参数 !!!!!!!!!!!!!!!!
# 最终模型的保存路径：
#  - 模型：{output_dir}/{task_name}/{checkpoints}/...
#  - 结果：{output_dir}/{task_name}/{log}/*.yaml

seed=${1:-0}

dataset_base='/data/Blob_EastUS/v-zhenwzhang/tensor_ts_datasets/Processed_Data/'
output_dir='/data/Blob_EastUS/v-zhenwzhang/log/tensor_ts_log/20241110/'
task_name='Main'

# His & Pred 搜索范围
HisLenList=(12 48 96 256)
PredLenList=(1 3 6 12 24 48)

# 模型列表
TensorModelList=("DCRNN" "NET3" "GraphWaveNet" "AGCRN" "MTGNN" "ST_Norm" "TTS_Norm" "GRML")
MultivarModelList=("TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "Mamba")

# 数据集：traffic, weather, finance, energy 
for his_len in ${HisLenList[@]}; do
    for pred_len in ${PredLenList[@]}; do
        # -------------------------------------------    
        # Tensor Model
        for model in ${TensorModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed
        done
        # -------------------------------------------
        # Multivar Model
        # for model in ${MultivarModelList[@]}; do
        #     # echo "$model, $his_len, $pred_len"
        #     bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base
        # done
        # -------------------------------------------
    done
done
