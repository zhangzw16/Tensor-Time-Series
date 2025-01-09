# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

seed=67

source $SCRIPT_DIR/Mode-0-TTS-Nature.sh $seed

source $SCRIPT_DIR/Mode-1-TTS-Nature.sh $seed

seed=109

source $SCRIPT_DIR/Mode-0-TTS-Nature.sh $seed

source $SCRIPT_DIR/Mode-1-TTS-Nature.sh $seed