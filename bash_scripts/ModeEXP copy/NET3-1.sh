# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="NET3-IS-${seed}"
data_mode=0
model='NET3'

dataset='JONAS_NYC_taxi'
# ====== Task0 ======
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dataset='METRO_HZ'
# ====== Task0 ======
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

dataset='COVID_CHI'
# ====== Task0 ======
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task0 ======
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================