# https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
import onnxruntime
import urllib.request
import cv2
import numpy as np
import json
#from onnx import numpy_helper
import time
from PIL import Image

#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.resnet50 import preprocess_input

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    #for i in range(img_data.shape[0]):
    #    norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    for i in range(img_data.shape[-1]):
        norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    #norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
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
    #onnx_model_path = "resnet50v2.onnx"
    onnx_model_path = "mobilenetv2-1.0.onnx"
    #onnx_model_path = "opt-efficientnet-b1.onnx"
    #onnx_model_path = "model.onnx"
    # onnx_model_path = "efficientnet_b0_experiments/output_model_baseline.onnx"
    
    imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    urllib.request.urlretrieve(imagenet_labels_url, filename="imagenet-simple-labels.json")
    label_map = load_labels("imagenet-simple-labels.json")
    
    sess = onnxruntime.InferenceSession(onnx_model_path)
    
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
    img = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = np.moveaxis(im_rgb, -1, 0)
    #img = np.expand_dims(img, axis=0)
    
    input_data = preprocess(im_rgb)
    input_data = np.moveaxis(input_data, -1, 1)
    print(img.shape)
    
    #img_path = 'dog.jpg'   # make sure the image is in img_path
    #img_size = 224
    #img = image.load_img(img_path, target_size=(img_size, img_size))
    #x = image.img_to_array(img)
    #x = np.expand_dims(x, axis=0)
    #input_data = preprocess_input(x)
    #print(input_data.shape)
    
    
    #input_data = img.copy()
    #for i in range(len(img)):
    #    input_data[i] = preprocess(img[i])

    """
    image = Image.open('dog.jpg')
    image_data = np.array(image).transpose(2, 0, 1)
    image_data1 = np.array(image)
    #print(image_data.shape)
    input_data = preprocess(image_data)
    
    cv2.imshow('displaymywindows', image_data1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
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
    