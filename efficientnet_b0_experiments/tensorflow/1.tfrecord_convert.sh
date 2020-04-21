set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

python imagenet_to_gcs.py \
--local_scratch_dir="tfrecord" \
--raw_data_dir="D:\Datasets\NB-CONN\divided" \
--file_format="png"
