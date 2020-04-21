# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
import onnxruntime
import urllib.request
import numpy as np
import json
import time
from PIL import Image

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
    onnx_model_path = "pytorch/efficientnet-b1.onnx"
    
    #imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    #urllib.request.urlretrieve(imagenet_labels_url, filename="imagenet-simple-labels.json")
    label_map = load_labels("imagenet-simple-labels.json")
    
    sess = onnxruntime.InferenceSession(onnx_model_path)
    sess.set_providers(['CPUExecutionProvider'])
    #print(onnxruntime.get_device())
    
    # input
    input_name = sess.get_inputs()[0].name
    print("Input name  :", input_name)
    input_shape = sess.get_inputs()[0].shape
    print("Input shape :", input_shape)
    input_type = sess.get_inputs()[0].type
    print("Input type  :", input_type)
    
    # output
    output_name = sess.get_outputs()[0].name
    print("Output name  :", output_name)  
    output_shape = sess.get_outputs()[0].shape
    print("Output shape :", output_shape)
    output_type = sess.get_outputs()[0].type
    print("Output type  :", output_type)
    

    img_path = "dog.jpg"
    img = Image.open(img_path)
    img = np.uint8(img)
    input_data = preprocess(img)
    input_data = np.moveaxis(input_data, -1, 1)
    print(input_data.shape)
    
    start = time.time()
    outputs = sess.run([output_name], {input_name: input_data})
    end = time.time()
    inference_time = np.round((end - start) * 1000, 2)
    
    res = postprocess(outputs)
    idx = np.argmax(res)
    print('========================================')
    print('Final top prediction is: ' + label_map[idx])
    print('========================================')
    
    print('========================================')
    print('Inference time: ' + str(inference_time) + " ms")
    print('========================================')
    
    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print('============ Top 5 labels are: ============================')
    print(label_map[sort_idx[:5]])
    print('===========================================================')
    