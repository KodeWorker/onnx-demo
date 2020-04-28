import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import efficientnet.tfkeras as efn
import tensorflow.compat.v1 as tf 
tf.disable_eager_execution()
import numpy as np
from PIL import Image
import json
import time

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')
    h = input_data.shape[0]
    w = input_data.shape[1]
    
    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[-1]):
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, h, w, 3).astype('float32')
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
    n_times = 100
    img_path = r"C:\Users\Tina_VI01\Desktop\KelvinWu\cat.1.jpg"
    label_map = load_labels("../imagenet-simple-labels.json")
     
    for model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']:
        
        x = Image.open(img_path)
        if model_name == 'efficientnet-b0':
            model = efn.EfficientNetB0(weights='imagenet')
            x = x.resize((224, 224))
        elif model_name == 'efficientnet-b1':
            model = efn.EfficientNetB1(weights='imagenet')
            x = x.resize((240, 240))
        elif model_name == 'efficientnet-b2':
            model = efn.EfficientNetB2(weights='imagenet')
            x = x.resize((260, 260))
        elif model_name == 'efficientnet-b3':
            model = efn.EfficientNetB3(weights='imagenet')
            x = x.resize((300, 300))
        elif model_name == 'efficientnet-b4':
            model = efn.EfficientNetB4(weights='imagenet')
            x = x.resize((380, 380))
        elif model_name == 'efficientnet-b5':
            model = efn.EfficientNetB5(weights='imagenet')
            x = x.resize((456, 456))
        elif model_name == 'efficientnet-b6':
            model = efn.EfficientNetB6(weights='imagenet')
            x = x.resize((528, 528))
        elif model_name == 'efficientnet-b7':
            model = efn.EfficientNetB7(weights='imagenet')
            x = x.resize((600, 600))
        x = preprocess(np.uint8(x))
        
        t1 = 0
        for _ in range(n_times):
            t0 = time.time()
            outputs = model.predict(x)
            t1 += time.time() - t0
        print("[{}] Elapsed Time: {:4f} sec.".format(model_name, t1/n_times))
        
        # res = postprocess(outputs)
        res = outputs.flatten()
        idx = np.argmax(res)
        
        print('========================================')
        print('Final top prediction is: ' + label_map[idx])
        print('========================================')
        
        sort_idx = np.flip(np.squeeze(np.argsort(res)))
        print("top 5 prob.: {}".format(np.array(res)[sort_idx[:5]]))
        print(label_map[sort_idx[:5]])