# !!!!!!!!!!!!! 需要设置下面三个参数 !!!!!!!!!!!!!!!!
# 最终模型的保存路径：
#  - 模型：{output_dir}/{task_name}/{checkpoints}/...
#  - 结果：{output_dir}/{task_name}/{log}/*.yaml

seed=${1:-0}

dataset_base='/data/Blob_EastUS/v-zhenwzhang/tensor_ts_datasets/Processed_Data/'
output_dir='/data/Blob_EastUS/v-zhenwzhang/log/tensor_ts_log/20241110/'
task_name="Main${seed}"
# 参数
normalizer='sklearn'        # 使用 sklearn 的 standard scaler
logger='tensorboard'        # ['none', 'wandb', 'tensorboard']
graph_init='pearson'        # 后续可能会用到，先留着

# 模型列表
ModelList=("DCRNN")

# 因为 Nature 数据集的长度太小（大约在200个点左右），单独遍历
NatureHisLenList=(6 12 24)
NaturePredLenList=(1 3 6 12)
NatureBatchSize=1   
for his_len in ${NatureHisLenList[@]}; do
    for pred_len in ${NaturePredLenList[@]}; do
        # -------------------------------------------    
        # Tensor Model
        for model in ${ModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_nature.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
    done
done

# His & Pred 搜索范围
HisLenList=(12 48 96 256)
PredLenList=(6 12 24 48)
batch_size=0                # 设置 0 开启 AutoBatch
# 数据集：traffic, weather, finance, energy
for his_len in ${HisLenList[@]}; do
    for pred_len in ${PredLenList[@]}; do
        # -------------------------------------------
        for model in ${ModelList[@]}; do            
            bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            # bash bash_scripts/run_energy.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            # bash bash_scripts/run_finance.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
    done
done