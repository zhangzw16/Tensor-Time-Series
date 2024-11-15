# !!!!!!!!!!!!! 需要设置下面三个参数 !!!!!!!!!!!!!!!!
# 最终模型的保存路径：
#  - 模型：{output_dir}/{task_name}/{checkpoints}/...
#  - 结果：{output_dir}/{task_name}/{log}/*.yaml
dataset_base='./datasets/data'
output_dir='./logs'
task_name='beautiful_weekend'
seed=2024
# 可选参数
graph_init='pearson' # only meaningful for graph-based models, ['pearson', 'inverse_pearson', 'random', 'cosine', 'unit']
data_mode=0 
batch_size=256       # AutoBatch is enabled in default.
normalizer='sklearn' # ['sklearn', 'std', 'none'], sklearn is for MTS, std is for Tensor.
logger='tensorboard'        # ['none', 'wandb', 'tensorboard']
# His & Pred 搜索范围
# HisLenList=(12 48 96 256)
# PredLenList=(1 3 6 12 24 48)
HisLenList=(12)
PredLenList=(12)

# 模型列表
# TensorModelList=("NET3" "DCRNN" "AGCRN" "MTGNN" "ST_Norm" "TTS_Norm" "GRML" "GCGRU" "Mamba" "STC_GNN" "GraphWaveNet")
# MultivarModelList=("TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "STID" "STWA")
# TensorModelList=("MTGNN" "ST_Norm" "TTS_Norm" "GRML" "GCGRU" "Mamba" )  # faster
TensorModelList=("NET3" "DCRNN" "AGCRN" "STC_GNN" "GraphWaveNet")       # slower
# MultivarModelList=("TimesNet" "StemGNN" "AutoFormer" "CrossFormer" "PatchTST" "DLinear" "NLinear" "STID")   # faster
MultivarModelList=("STWA") # slower


# 数据集：traffic, weather, finance, energy 
for his_len in ${HisLenList[@]}; do
    for pred_len in ${PredLenList[@]}; do
        # -------------------------------------------    
        # Tensor Model
        for model in ${TensorModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_finance.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_energy.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
        # -------------------------------------------
        # Multivar Model
        for model in ${MultivarModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_traffic.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_weather.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_finance.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
            bash bash_scripts/run_energy.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
        # -------------------------------------------
    done
done

# 因为 Nature 数据集的长度太小（大约在200个点左右），单独遍历
NatureHisLenList=(6 12 24)
NaturePredLenList=(1 3 6 12)
NatureBatchSize=1   
for his_len in ${NatureHisLenList[@]}; do
    for pred_len in ${NaturePredLenList[@]}; do
        # -------------------------------------------    
        # Tensor Model
        for model in ${TensorModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_nature.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
        # -------------------------------------------
        # Multivar Model
        for model in ${MultivarModelList[@]}; do
            # echo "$model, $his_len, $pred_len"
            bash bash_scripts/run_nature.sh --model $model --his_len $his_len --pred_len $pred_len --output_dir $output_dir --task_name $task_name --dataset_base $dataset_base --seed $seed --graph_init $graph_init --data_mode $data_mode --batch_size $batch_size --logger $logger --normalizer $normalizer
        done
        # -------------------------------------------
    done
done
