set -e
CONDA_PATH=$(where anaconda)
CONDA_SETTING_SCRIPT="${CONDA_PATH}/../../etc/profile.d/conda.sh"
source "${CONDA_SETTING_SCRIPT}"
conda activate TensorFlow2

python tpu_efficientnet/eval_ckpt_main.py \
--model_name="efficientnet-b0" \
--ckpt_dir="./pretrained_AA/efficientnet-b0" \
--example_img="D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg" \
--labels_map_file="label_map.txt" \
--runmode="examples"

#python tpu_efficientnet/eval_ckpt_main.py \
#--model_name="efficientnet-b0" \
#--ckpt_dir="./EfficentNet-b0/archive" \
#--example_img="D:\Datasets\NB-CONN\divided\validation\NG\NB1_14.png" \
#--labels_map_file="label_map.txt" \
#--runmode="examples"