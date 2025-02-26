
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh
export output_dir='/nas/datasets/zjx/datasets/TensorTSL_Output/1e-4/'
# 命令行传入 seed
seed=${1:-0}
task_name="ModeEXP-TTS-1e-4-${seed}"
data_mode=1


# ====== Task0 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task1 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task2 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task3 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task4 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task5 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task6 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task7 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task8 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task9 ======
dataset='weather'
model='GraphWaveNet'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task10 ======
dataset='weather'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task11 ======
dataset='weather'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================
