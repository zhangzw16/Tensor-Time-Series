
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="ModeEXP-MTS-Nature-${seed}"
data_mode=0


# ====== Task0 ======
dataset='COVID_DEATHS'
model='DLinear'
his_len=6
pred_len=6
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task1 ======
dataset='COVID_DEATHS'
model='DLinear'
his_len=6
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task2 ======
dataset='COVID_DEATHS'
model='DLinear'
his_len=6
pred_len=24
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task3 ======
dataset='COVID_DEATHS'
model='STID'
his_len=6
pred_len=6
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task4 ======
dataset='COVID_DEATHS'
model='STID'
his_len=6
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task5 ======
dataset='COVID_DEATHS'
model='STID'
his_len=6
pred_len=24
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task6 ======
dataset='COVID_DEATHS'
model='TimesNet'
his_len=6
pred_len=6
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task7 ======
dataset='COVID_DEATHS'
model='TimesNet'
his_len=6
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task8 ======
dataset='COVID_DEATHS'
model='TimesNet'
his_len=6
pred_len=24
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task9 ======
dataset='COVID_DEATHS'
model='PatchTST'
his_len=12
pred_len=6
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task10 ======
dataset='COVID_DEATHS'
model='PatchTST'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task11 ======
dataset='COVID_DEATHS'
model='PatchTST'
his_len=12
pred_len=24
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================
