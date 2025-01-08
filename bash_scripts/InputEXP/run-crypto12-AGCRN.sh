# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="InputSearch-v2-${seed}"
data_mode=0

dataset='crypto12'
model='AGCRN'

# ====== Task1 ======
his_len=1
pred_len=12
batch_size=0    # 设置 0 开启 AutoBatch
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task1 ======
his_len=3
pred_len=12
batch_size=0    # 设置 0 开启 AutoBatch
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task1 ======
his_len=6
pred_len=12
batch_size=0    # 设置 0 开启 AutoBatch
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task1 ======
his_len=9
pred_len=12
batch_size=0    # 设置 0 开启 AutoBatch
source $SCRIPT_DIR/run_task.sh
# ===================

# ====== Task1 ======
his_len=12
pred_len=12
batch_size=0    # 设置 0 开启 AutoBatch
source $SCRIPT_DIR/run_task.sh
# ===================