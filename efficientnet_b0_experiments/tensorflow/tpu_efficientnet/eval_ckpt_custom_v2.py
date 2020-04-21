import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import sys
import eval_ckpt_main as eval_ckpt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import time

if __name__ == "__main__":
    labels_map_file = '../label_map.txt'
    model_name = "efficientnet-b0"
    ckpt_dir = "../pretrained_AA/efficientnet-b0"
    
    image_files = [r"D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg"] # *.png not compatible
    eval_driver = eval_ckpt.get_eval_driver(model_name)
    
    t0 = time.time()
    pred_idx, pred_prob = eval_driver.eval_example_images(ckpt_dir, image_files, labels_map_file)
    print("Elapsed Time: {:4f} sec.".format(time.time() - t0))