SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

seed=42

data_mode=0
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

data_mode=1
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

# data_mode=2
# source $SCRIPT_DIR/HM.sh $seed $data_mode

# data_mode=3
# source $SCRIPT_DIR/HM.sh $seed $data_mode


seed=67

data_mode=0
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

data_mode=1
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

# data_mode=2
# source $SCRIPT_DIR/HM.sh $seed $data_mode

# data_mode=3
# source $SCRIPT_DIR/HM.sh $seed $data_mode


seed=109

data_mode=0
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

data_mode=1
source $SCRIPT_DIR/HM_TTS.sh $seed $data_mode

# data_mode=2
# source $SCRIPT_DIR/HM.sh $seed $data_mode

# data_mode=3
# source $SCRIPT_DIR/HM.sh $seed $data_mode