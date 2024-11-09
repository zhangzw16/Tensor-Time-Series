# --------------- Check Directory -------------------
# get the current directory
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# get the parent directory
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# check if the current directory is the parent directory, if not, switch to the directory
if [ "$PWD" != "$PARENT_DIR" ]; then
    echo "Changing directory to $PARENT_DIR"
    cd "$PARENT_DIR" || exit
else
    echo "Already in the parent directory: $PARENT_DIR"
fi

# ---------------- Task Config ----------------------
# 1. Basic configuration
task_name="Main"
output_dir='./output'
train_test='train'
device='cuda'
logger='none'

# 2. Dataset configuration
batch_size=128
his_len=96
pred_len=12
data_mode=0

# 3. optional
normalizer='std'
graph_init='pearson'
scheduler='ReduceLROnPlateau'
# ----------------------------------------------------

export CUDA_VISIBLE_DEVICES=0

# --------------- Select Data & Model ----------------
# >>> Run 1 >>>>>>>>>>>>>
dataset='JONAS_NYC_bike'
model='TimesNet'
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
# # >>> Run 2 >>>>>>>>>>>>>
# dataset='weather'
# model='TimesNet'
# python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
#                     --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
#                     --normalizer $normalizer --graph_init $graph_init \
#                     --scheduler $scheduler \
#                     --lr_finder \
#                     --logger $logger \
#                     --dataset $dataset --model $model \
# >>> Run 3 >>>>>>>>>>>>>
# dataset='ETT_hour'
# model='TimesNet'
# python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
#                     --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
#                     --normalizer $normalizer --graph_init $graph_init \
#                     --scheduler $scheduler \
#                     --lr_finder \
#                     --logger $logger \
#                     --dataset $dataset --model $model \
# # >>> Run 4 >>>>>>>>>>>>>
# dataset='electricity'
# model='TimesNet'
# python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
#                     --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
#                     --normalizer $normalizer --graph_init $graph_init \
#                     --scheduler $scheduler \
#                     --lr_finder \
#                     --logger $logger \
#                     --dataset $dataset --model $model \
