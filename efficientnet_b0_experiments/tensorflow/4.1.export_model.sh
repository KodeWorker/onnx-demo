set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

MODEL_NAME="efficientnet-b0"
CKPT_DIR="./efficientnet-b0-baseline"
OUTPUT_TFLITE="tflite_model"
QUANTIZE=false
OUTPUT_SAVED_MODEL_DIR="./output_model_baseline"
NUM_CLASSES=1000
#DATA_DIR="./tfrecord"

python tpu_efficientnet/export_model.py \
--model_name=$MODEL_NAME \
--ckpt_dir=$CKPT_DIR \
--output_tflite=$OUTPUT_TFLITE \
--quantize=$QUANTIZE \
#--data_dir=$DATA_DIR \
--output_saved_model_dir=$OUTPUT_SAVED_MODEL_DIR \
--num_classes=$NUM_CLASSES