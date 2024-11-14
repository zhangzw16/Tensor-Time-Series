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
task_name="2024-11-13"
output_dir='./logs'
train_test='train'
model_path=""
device='cuda'
logger='tensorboard'

# 2. Dataset configuration
batch_size=256  # AutoBach is enabled in default, this parameter will be ignored
his_len=12
pred_len=12
data_mode=0

# 3. optional
normalizer='std'
graph_init='pearson'
scheduler='ReduceLROnPlateau'
# ----------------------------------------------------

export CUDA_VISIBLE_DEVICES=2

# --------------- Select Data & Model ----------------
# >>> Run 1 >>>>>>>>>>>>>
dataset='PEMSBAY'
model='GraphWaveNet'
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
                    --debug \
                    # --model_path $model_path
