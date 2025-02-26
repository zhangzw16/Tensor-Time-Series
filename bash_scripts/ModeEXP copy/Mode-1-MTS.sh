
# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

# 命令行传入 seed
seed=${1:-0}
task_name="ModeEXP-MTS-${seed}"
data_mode=1


# ====== Task0 ======
dataset='JONAS_NYC_taxi'
model='DLinear'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task1 ======
dataset='JONAS_NYC_taxi'
model='DLinear'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task2 ======
dataset='JONAS_NYC_taxi'
model='DLinear'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task3 ======
dataset='COVID_CHI'
model='DLinear'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task4 ======
dataset='COVID_CHI'
model='DLinear'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task5 ======
dataset='COVID_CHI'
model='DLinear'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task6 ======
dataset='METRO_HZ'
model='DLinear'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task7 ======
dataset='METRO_HZ'
model='DLinear'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task8 ======
dataset='METRO_HZ'
model='DLinear'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task9 ======
dataset='ETT_hour'
model='DLinear'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task10 ======
dataset='ETT_hour'
model='DLinear'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task11 ======
dataset='ETT_hour'
model='DLinear'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task12 ======
dataset='weather'
model='DLinear'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task13 ======
dataset='weather'
model='DLinear'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task14 ======
dataset='weather'
model='DLinear'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task15 ======
dataset='crypto12'
model='DLinear'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task16 ======
dataset='crypto12'
model='DLinear'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task17 ======
dataset='crypto12'
model='DLinear'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task18 ======
dataset='stocknet'
model='DLinear'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task19 ======
dataset='stocknet'
model='DLinear'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task20 ======
dataset='stocknet'
model='DLinear'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task21 ======
dataset='JONAS_NYC_taxi'
model='STID'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task22 ======
dataset='JONAS_NYC_taxi'
model='STID'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task23 ======
dataset='JONAS_NYC_taxi'
model='STID'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task24 ======
dataset='COVID_CHI'
model='STID'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task25 ======
dataset='COVID_CHI'
model='STID'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task26 ======
dataset='COVID_CHI'
model='STID'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task27 ======
dataset='METRO_HZ'
model='STID'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task28 ======
dataset='METRO_HZ'
model='STID'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task29 ======
dataset='METRO_HZ'
model='STID'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task30 ======
dataset='ETT_hour'
model='STID'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task31 ======
dataset='ETT_hour'
model='STID'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task32 ======
dataset='ETT_hour'
model='STID'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task33 ======
dataset='weather'
model='STID'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task34 ======
dataset='weather'
model='STID'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task35 ======
dataset='weather'
model='STID'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task36 ======
dataset='crypto12'
model='STID'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task37 ======
dataset='crypto12'
model='STID'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task38 ======
dataset='crypto12'
model='STID'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task39 ======
dataset='stocknet'
model='STID'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task40 ======
dataset='stocknet'
model='STID'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task41 ======
dataset='stocknet'
model='STID'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task42 ======
dataset='JONAS_NYC_taxi'
model='TimesNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task43 ======
dataset='JONAS_NYC_taxi'
model='TimesNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task44 ======
dataset='JONAS_NYC_taxi'
model='TimesNet'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task45 ======
dataset='COVID_CHI'
model='TimesNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task46 ======
dataset='COVID_CHI'
model='TimesNet'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task47 ======
dataset='COVID_CHI'
model='TimesNet'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task48 ======
dataset='METRO_HZ'
model='TimesNet'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task49 ======
dataset='METRO_HZ'
model='TimesNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task50 ======
dataset='METRO_HZ'
model='TimesNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task51 ======
dataset='ETT_hour'
model='TimesNet'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task52 ======
dataset='ETT_hour'
model='TimesNet'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task53 ======
dataset='ETT_hour'
model='TimesNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task54 ======
dataset='weather'
model='TimesNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task55 ======
dataset='weather'
model='TimesNet'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task56 ======
dataset='weather'
model='TimesNet'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task57 ======
dataset='crypto12'
model='TimesNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task58 ======
dataset='crypto12'
model='TimesNet'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task59 ======
dataset='crypto12'
model='TimesNet'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task60 ======
dataset='stocknet'
model='TimesNet'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task61 ======
dataset='stocknet'
model='TimesNet'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task62 ======
dataset='stocknet'
model='TimesNet'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task63 ======
dataset='JONAS_NYC_taxi'
model='PatchTST'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task64 ======
dataset='JONAS_NYC_taxi'
model='PatchTST'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task65 ======
dataset='JONAS_NYC_taxi'
model='PatchTST'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task66 ======
dataset='COVID_CHI'
model='PatchTST'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task67 ======
dataset='COVID_CHI'
model='PatchTST'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task68 ======
dataset='COVID_CHI'
model='PatchTST'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task69 ======
dataset='METRO_HZ'
model='PatchTST'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task70 ======
dataset='METRO_HZ'
model='PatchTST'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task71 ======
dataset='METRO_HZ'
model='PatchTST'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task72 ======
dataset='ETT_hour'
model='PatchTST'
his_len=96
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task73 ======
dataset='ETT_hour'
model='PatchTST'
his_len=96
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task74 ======
dataset='ETT_hour'
model='PatchTST'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task75 ======
dataset='weather'
model='PatchTST'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task76 ======
dataset='weather'
model='PatchTST'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task77 ======
dataset='weather'
model='PatchTST'
his_len=96
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task78 ======
dataset='crypto12'
model='PatchTST'
his_len=12
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task79 ======
dataset='crypto12'
model='PatchTST'
his_len=12
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task80 ======
dataset='crypto12'
model='PatchTST'
his_len=12
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task81 ======
dataset='stocknet'
model='PatchTST'
his_len=48
pred_len=12
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task82 ======
dataset='stocknet'
model='PatchTST'
his_len=48
pred_len=48
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================


# ====== Task83 ======
dataset='stocknet'
model='PatchTST'
his_len=48
pred_len=96
batch_size=0
source $SCRIPT_DIR/run_task.sh
# ===================
