# run task list in a GPU
export CUDA_VISIBLE_DEVICES=0
data_mode=0
task_name="MTS_Task"
only_test="False"
output_dir="./output"
# ['Traffic', 'Natural', 'Energy', 'Weather', 'Finance']
# data_mode = 0 (1, time, dim1*dim2, 1)
# data_mode = 1 (dim2, time, dim1, 1)
# data_mode = 2 (dim1, time, dim2, 1)

status_file="./output/status/${task_name}_${data_mode}.txt"
mkdir -p "$(dirname "$status_file")"
touch $status_file

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start at [${timestamp}]" > $status_file
python3 run_tasks.py --his_len 96 \
    --pred_len 12 --dataset Finance \
    --task_name ${task_name} --output_dir $output_dir \
    --data_mode $data_mode --only_test $only_test