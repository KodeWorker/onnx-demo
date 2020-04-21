set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

# save keras model as savedMoel

#python export_pretrained_efficientnet_b0.py

# show input/output names

#DIR="./saved_model"
#TAG_SET="serve"
#DIGNATURE_DEF="serving_default"

#saved_model_cli show \
#--dir $DIR \
#--tag_set $TAG_SET \
#--signature_def $DIGNATURE_DEF

# convert savedModel to onnx

#SOURCE_SAVED_MODEL="saved_model"
#TARGET_ONNX_MODEL="tfkeras_pretrained.onnx"
#INPUTS="serving_default_input_1:0"
#OUTPUTS="StatefulPartitionedCall:0"
#OPSET=11

#python -m tf2onnx.convert \
#    --saved-model $SOURCE_SAVED_MODEL \
#    --output $TARGET_ONNX_MODEL \
#	--inputs $INPUTS \
#    --outputs $OUTPUTS \
#	--opset $OPSET \

SOURCE_GRAPHDEF_PB="./export/frozen_model.pb"
TARGET_ONNX_MODEL="tfkeras_pretrained.onnx"
INPUTS="input_1:0"
OUTPUTS="probs/Softmax:0"
OPSET=11

python -m tf2onnx.convert \
    --input $SOURCE_GRAPHDEF_PB \
    --output $TARGET_ONNX_MODEL \
	--inputs $INPUTS \
    --outputs $OUTPUTS \
	--opset $OPSET \

