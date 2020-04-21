import cv2
import json
import numpy as np
import tensorflow as tf

def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    #mean_vec = np.array([0, 0, 0])
    #stddev_vec = np.array([1, 1, 1])
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

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path="pretrained_efficientnet_b0.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    
    label_map = load_labels("../../imagenet-simple-labels.json")
    img_path = "cat.1.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = preprocess(im_rgb)
    print("input_data shape: {}".format(input_data.shape))
    
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    #res = postprocess(output_data)
    res = output_data[0]
    idx = np.argmax(res)
    print('========================================')
    print('Final top prediction is: ' + label_map[idx])
    print('========================================')
    
    
    sort_idx = np.flip(np.squeeze(np.argsort(res)))
    print('============ Top 5 labels are: ============================')
    for idx in sort_idx[:5]:
        print("[{}]({:.2f}%): {}".format(idx, res[idx]*100, label_map[idx]))
    print('===========================================================')