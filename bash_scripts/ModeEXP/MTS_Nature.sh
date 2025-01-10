# 获取环境变量
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source $SCRIPT_DIR/env.sh

seed=${1:-0}

source $SCRIPT_DIR/Mode-0-MTS-Nature.sh $seed

source $SCRIPT_DIR/Mode-1-MTS-Nature.sh $seed

source $SCRIPT_DIR/Mode-2-MTS-Nature.sh $seed

source $SCRIPT_DIR/Mode-3-MTS-Nature.sh $seed
