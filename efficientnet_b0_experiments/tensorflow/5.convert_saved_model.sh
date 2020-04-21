set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

SOURCE_SAVED_MODEL=output_model_baseline
TARGET_ONNX_MODEL=output_model_baseline.onnx

python -m tf2onnx.convert \
    --saved-model $SOURCE_SAVED_MODEL \
    --output $TARGET_ONNX_MODEL \
	--inputs images:0 \
    --outputs Softmax:0 \
	--opset 11