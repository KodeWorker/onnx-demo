# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from PIL import Image
import json

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[-1]):
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 224, 224, 3).astype('float32')
    return norm_img_data

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def postprocess(result):
    return softmax(np.array(result)).tolist()

if __name__ == "__main__":
    # img_path = "../dog.jpg"
    img_path = r"D:\Datasets\Kaggle\dogs-vs-cats\train\cat\cat.1.jpg"
    save_dir = "./efficientnet_saved_model"
    label_map = load_labels("../imagenet-simple-labels.json")
    
    #with tf.device("/cpu:0"):
    model = tf.saved_model.load(save_dir)
    f = model.signatures["serving_default"]
    
    x = Image.open(img_path)
    x = x.resize((224, 224))
    x = preprocess(np.uint8(x))
    x = np.moveaxis(x, -1, 1)
    y = f(input=tf.constant(np.float32(x)))
    outputs = y["output"].numpy()
    
    res = postprocess(outputs)
    idx = np.argmax(res)
    
    print('========================================')
    print('Final top prediction is: ' + label_map[idx])
    print('========================================')
    
    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print("top 5 prob.: {}".format(np.array(res)[sort_idx[:5]]))
    print(label_map[sort_idx[:5]])