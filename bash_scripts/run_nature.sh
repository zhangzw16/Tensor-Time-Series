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
batch_size=1
his_len=12
pred_len=12
data_mode=0                     # it depends on the model_type, more details can be found in '--help'

# 3. optional
normalizer='std'
graph_init='pearson'            # only meaningful for graph-based models
scheduler='ReduceLROnPlateau'   # more detailed settings can be found in 'utils/scheduler/schedulerManager.py'
# ----------------------------------------------------

export CUDA_VISIBLE_DEVICES=0

# --------------- Select Data & Model ----------------
# start training with the following configurations
# "Nature": ['COVID_DEATHS', 'COVID_CHI', 'COVID_US'],
# get the model name from the command line
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) model="$2"; shift ;;
        --his_len) his_len="$2"; shift ;;
        --pred_len) pred_len="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --task_name) task_name="$2"; shift ;;
        --data_base) data_base="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# check if the model name is specified
if [ -z "$model" ] || [ -z "$his_len" ] || [ -z "$pred_len" ]; then
    echo "Usage: $0 --model <model> --his_len <his_len> --pred_len <pred_len>"
    exit 1
fi
# >>>> DS1
dataset='COVID_DEATHS'
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --data_base $data_base \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
# >>>> DS2
dataset='COVID_CHI'
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --data_base $data_base \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
# >>>> DS3
dataset='COVID_US'
python3 main_cli.py --task_name $task_name --output_dir $output_dir --train_test $train_test --device $device \
                    --batch_size $batch_size --his_len $his_len --pred_len $pred_len --data_mode $data_mode \
                    --normalizer $normalizer --graph_init $graph_init \
                    --data_base $data_base \
                    --scheduler $scheduler \
                    --lr_finder \
                    --logger $logger \
                    --dataset $dataset --model $model \
