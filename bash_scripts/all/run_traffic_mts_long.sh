# !!!!!!!!!!!!! 需要设置下面三个参数 !!!!!!!!!!!!!!!!
# 最终模型的保存路径：
#  - 模型：{output_dir}/{task_name}/{checkpoints}/...
#  - 结果：{output_dir}/{task_name}/{log}/*.yaml

seed=${1:-0}

dataset_base='/data/Blob_EastUS/v-zhenwzhang/tensor_ts_datasets/Processed_Data/'
output_dir='/data/Blob_EastUS/v-zhenwzhang/log/tensor_ts_log/20241110/'
task_name='Main'${seed}

# 参数
normalizer='sklearn'        # 使用 sklearn 的 standard scaler
logger='tensorboard'        # ['none', 'wandb', 'tensorboard']
batch_size=256              # 默认开启 AutoBatch，因此这个不起作用
graph_init='pearson'        # 后续可能会用到，先留着

# His & Pred 搜索范围
HisLenList=(12 48 96 256)
PredLenList=(24 48)

# 模型列表
TensorModelList=("DCRNN" "NET3" "AGCRN" "MTGNN" "ST_Norm" "TTS_Norm" "GRML" "GraphWaveNet")
MultivarModelList=("TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "Mamba")

# 数据集：traffic, weather, finance, energy 
for his_len in ${HisLenList[@]}; do
    for pred_len in ${PredLenList[@]}; do
        # -------------------------------------------    
        # Tensor Model
        # for model in ${TensorModelList[@]}; do
        #     # echo "$model, $his_len, $pred_len"
        #     bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base
        # done
        # -------------------------------------------
        # Multivar Model
        for model in ${MultivarModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
        # -------------------------------------------
    done
done
