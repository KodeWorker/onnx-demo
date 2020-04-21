import cntk as C
import numpy as np
from PIL import Image
import pickle
import json 

def load_labels(path):
    with open(path) as f:
        data = json.load(f)
    return np.asarray(data)
    
def preprocess(input_data):
    # convert the input data into the float32 input
    img_data = input_data.astype('float32')

    #normalize
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
    #for i in range(img_data.shape[-1]):
    #    norm_img_data[:,:,i] = (img_data[:,:,i]/255 - mean_vec[i]) / stddev_vec[i]
        
    #add batch channel
    #norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    #norm_img_data = norm_img_data.reshape(1, 224, 224, 3).astype('float32')
    return norm_img_data
    
# Import the model into CNTK via the CNTK import API
z = C.Function.load("resnet50v2.onnx", device=C.device.cpu(), format=C.ModelFormat.ONNX)
print("Loaded resnet50v2.onnx!")
img = Image.open("dog.jpg")
img = img.resize((224,224))
img = np.array(img).transpose(2, 0, 1)
img = np.array(img)
img_data = preprocess(img)

predictions = np.squeeze(z.eval({z.arguments[0]:[img_data]}))
top_class = np.argmax(predictions)
print(top_class)
label_map = load_labels("imagenet-simple-labels.json")
print(label_map[top_class])