#!/bin/bash

# !!!! 说明:
# 1. 其中 Stat 和 MultiVars 的模型可以放在一张或者两张卡上跑，跑的挺快的
# 2. 对于 Tensor 模型， Tensor-prior 模型跑的超级慢，建议每个模型放单独一张卡上跑，其中 DCRNN 对于内存的需求比较大
# 3. !!! 对于 Natural 数据而言，由于数据集本身比较小，所以 batch size 设置在 4 左右比较合适，不然会因为数据集不够而报错

# 设定参数
his_len=16
pred_len=16
batch_size=16
# ['Stat', 'MultiVar', 'Tensor-prior', 'Tensor-learned', 'Tensor-none']
model_type='Tensor-prior'
# 默认是空字符。 设定 model_type 后自动找到对应的模型。
# 该参数是用于特定模型的补充实验
model_name=''
# ['Traffic', 'Natural', 'Energy'] 
dataset_type='Traffic'
# 0: 默认的 (time, dim1, dim2)， 1: 压成一维 (time, dim1 * dim2, 1)
data_mode=0
# 选择 checkpoint 和 log(.yaml) 输出的目录
output_dir='./output'

# 参数解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --his_len)
            his_len="$2"
            shift # 移过选项
            shift # 移过选项的值
            ;;
        --pred_len)
            pred_len="$2"
            shift
            shift
            ;;
        --batch_size)
            batch_size="$2"
            shift
            shift
            ;;
        --model_type)
            model_type="$2"
            shift
            shift
            ;;
        --model_name)
            model_name="$2"
            shift
            shift
            ;;
        --dataset_type)
            dataset_type="$2"
            shift
            shift
            ;;
        --data_mode)
            data_mode="$2"
            shift
            shift
            ;;
        --output_dir)
            output_dir="$2"
            shift
            shift
            ;;
        *)    # 对于不认识的选项，直接忽略或者给出错误提示
            echo "unknow args: $1"
            shift
            ;;
    esac
done


# 简单检查一下：
# echo "历史长度: $his_len"
# echo "预测长度: $pred_len"
# echo "批处理大小: $batch_size"
# echo "模型类型: $model_type"
# echo "模型名称: $model_name"
# echo "数据集类型: $dataset_type"
# echo "数据集模式: $dataset_mode"
# echo "输出目录: $output_dir"


# 冲冲冲
python3 shell_run.py --his_len $his_len --pred_len $pred_len \
        --batch_size $batch_size \
        --model_type $model_type --model_name $model_name \
        --dataset_type $dataset_type --data_mode $data_mode \
        --output_dir $output_dir