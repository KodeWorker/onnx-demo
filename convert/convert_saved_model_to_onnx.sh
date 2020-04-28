set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

for MODEL_NAME in "efficientnet-b0" "efficientnet-b1" "efficientnet-b2" "efficientnet-b3" "efficientnet-b4" "efficientnet-b5" "efficientnet-b6" "efficientnet-b7"
do

	SOURCE_SAVED_MODEL="tensorflow_models/pretrained-"$MODEL_NAME
	TARGET_ONNX_MODEL=$MODEL_NAME".onnx"

	python -m tf2onnx.convert \
		--saved-model $SOURCE_SAVED_MODEL \
		--output $TARGET_ONNX_MODEL \
		--inputs input:0 \
		--outputs output:0 \
		--opset 11
done