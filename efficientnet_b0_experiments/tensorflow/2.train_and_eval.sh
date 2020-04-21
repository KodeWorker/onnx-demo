set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

python tpu_efficientnet/main.py \
--data_dir="./tfrecord" \
--model_dir="./EfficentNet-b0" \
--model_name="efficientnet-b0" \
--skip_host_call=True \
--train_batch_size=8 \
--num_train_images=222 \
--num_eval_images=51 \
--train_steps=160 \
--use_tpu=False \
--data_format="channels_first" \
--transpose_input=False \
--mode="train_and_eval" \
--eval_batch_size=8 \
--num_label_classes=2 \
--steps_per_eval=10 \