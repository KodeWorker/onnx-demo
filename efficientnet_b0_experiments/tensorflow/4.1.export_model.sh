set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

MODEL_NAME="efficientnet-b5"
CKPT_DIR="./pretrained_model/noisy-student-"$MODEL_NAME
OUTPUT_TFLITE="./pretrained-"$MODEL_NAME".tflite"
QUANTIZE=false
OUTPUT_SAVED_MODEL_DIR="pretrained-"$MODEL_NAME
NUM_CLASSES=1000

rm -r -f $OUTPUT_SAVED_MODEL_DIR
mkdir $OUTPUT_SAVED_MODEL_DIR

python tpu_efficientnet/export_model.py \
--model_name=$MODEL_NAME \
--ckpt_dir=$CKPT_DIR \
--output_tflite=$OUTPUT_TFLITE \
--quantize=$QUANTIZE \
--output_saved_model_dir=$OUTPUT_SAVED_MODEL_DIR \
--num_classes=$NUM_CLASSES