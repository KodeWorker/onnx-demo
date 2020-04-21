set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

DIR="./output_model"
TAG_SET="serve"
DIGNATURE_DEF="serving_default"

saved_model_cli show \
--dir $DIR \
--tag_set $TAG_SET \
--signature_def $DIGNATURE_DEF