set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate openvino

MODEL_NAME="efficientnet-b7"
INPUT_MODEL="opt-"$MODEL_NAME".onnx"
OUTPUT_DIR=$MODEL_NAME

rm -r -f $MODEL_NAME
mkdir $OUTPUT_DIR

python "C:\Program Files (x86)\IntelSWTools\openvino\deployment_tools\model_optimizer\mo.py" \
--input_shape=[1,3,224,224] \
--data_type=FP16 \
--mean_values "[123.675, 116.28, 103.53]" \
--scale_values "[58.395, 57.12, 57.375]" \
--reverse_input_channels \
--input_model $INPUT_MODEL \
--output_dir $OUTPUT_DIR