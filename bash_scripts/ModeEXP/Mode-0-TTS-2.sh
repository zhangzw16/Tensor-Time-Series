
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="ModeEXP-TTS-2-${seed}"
data_mode=0


# ====== Task0 ======
dataset='JONAS_NYC_taxi'
model='AGCRN'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task1 ======
dataset='JONAS_NYC_taxi'
model='AGCRN'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task2 ======
dataset='JONAS_NYC_taxi'
model='AGCRN'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task3 ======
dataset='COVID_CHI'
model='AGCRN'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task4 ======
dataset='COVID_CHI'
model='AGCRN'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task5 ======
dataset='COVID_CHI'
model='AGCRN'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task6 ======
dataset='METRO_HZ'
model='AGCRN'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task7 ======
dataset='METRO_HZ'
model='AGCRN'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task8 ======
dataset='METRO_HZ'
model='AGCRN'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task9 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task10 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task11 ======
dataset='ETT_hour'
model='AGCRN'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task12 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task13 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task14 ======
dataset='weather'
model='AGCRN'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task15 ======
dataset='crypto12'
model='AGCRN'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task16 ======
dataset='crypto12'
model='AGCRN'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task17 ======
dataset='crypto12'
model='AGCRN'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task18 ======
dataset='stocknet'
model='AGCRN'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task19 ======
dataset='stocknet'
model='AGCRN'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task20 ======
dataset='stocknet'
model='AGCRN'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task21 ======
dataset='JONAS_NYC_taxi'
model='GraphWaveNet'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task22 ======
dataset='JONAS_NYC_taxi'
model='GraphWaveNet'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task23 ======
dataset='JONAS_NYC_taxi'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task24 ======
dataset='COVID_CHI'
model='GraphWaveNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task25 ======
dataset='COVID_CHI'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task26 ======
dataset='COVID_CHI'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task27 ======
dataset='METRO_HZ'
model='GraphWaveNet'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task28 ======
dataset='METRO_HZ'
model='GraphWaveNet'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task29 ======
dataset='METRO_HZ'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task30 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task31 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task32 ======
dataset='ETT_hour'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task33 ======
dataset='weather'
model='GraphWaveNet'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task34 ======
dataset='weather'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task35 ======
dataset='weather'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task36 ======
dataset='crypto12'
model='GraphWaveNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task37 ======
dataset='crypto12'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task38 ======
dataset='crypto12'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task39 ======
dataset='stocknet'
model='GraphWaveNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task40 ======
dataset='stocknet'
model='GraphWaveNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task41 ======
dataset='stocknet'
model='GraphWaveNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================
