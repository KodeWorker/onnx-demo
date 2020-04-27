set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate openvino

MODEL_NAME="efficientnet-b0"
INPUT_MODEL="opt-"$MODEL_NAME".onnx"

python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" \
--input_shape=[1,160,160,3] \
--data_type=FP16 \
--input_model $INPUT_MODEL \
--output_dir openvino_model