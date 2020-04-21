import os
import sys
import eval_ckpt_main as eval_ckpt
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    valdir = r"D:\Datasets\NB-CONN\20200326\divided\validation"
    labels_map_file = '../label_map.txt'
    model_name = "efficientnet-b0"
    ckpt_dir = "../EfficentNet-b0"
    
    dirs = os.listdir(valdir)
    
    for dir_ in dirs:
        filenames = os.listdir(os.path.join(valdir, dir_))
        #image_files = [os.path.join(valdir, dir_, name) for name in filenames]
        image_files = ["../NB1_14.jpg"] # *.png not compatible
        eval_driver = eval_ckpt.get_eval_driver(model_name)
        pred_idx, pred_prob = eval_driver.eval_example_images(ckpt_dir, image_files, labels_map_file)
        print(pred_idx)
