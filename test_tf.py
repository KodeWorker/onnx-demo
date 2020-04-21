import tensorflow as tf
from tensorflow.python.platform import gfile
import time
import numpy as np
import json
import cv2

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
        
    #add batch channel
    norm_img_data = norm_img_data.reshape(1, 3, 224, 224).astype('float32')
    return norm_img_data

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == "__main__":

    
    # MODEL_PB = './mobilenetv2-1.0-tf.pb'
    MODEL_PB = "./efficientnet_b0_experiments/keras/export/frozen_model.pb"
    OUTPUT_PATH = 'events/'
    graph = load_pb(MODEL_PB)
    
    img_size = 224
    num_channels = 3
    
    tensor_names = [t.name for op in graph.get_operations() for t in op.values()]
    for op in graph.get_operations():
        for t in op.values():
            # if "data" in t.name:
                print(t.name)
    
    label_map = load_labels("imagenet-simple-labels.json")
            
    img_path = "dog.jpg"
    img = cv2.imread(img_path)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.moveaxis(im_rgb, -1, 0)
    img = np.expand_dims(img, axis=0)
    input_data = preprocess(img)
    
    start = time.time()
    
    # input = graph.get_tensor_by_name("serving_default_input_5:0")
    # output = graph.get_tensor_by_name("StatefulPartitionedCall:0")
    
    # with tf.compat.v1.Session(graph=graph) as sess:
    #     predictions = sess.run(output, feed_dict={input: input_data})
        
    # end = time.time()
    # inference_time = np.round((end - start) * 1000, 2)
    
    # idx = np.argmax(predictions)
    
    # print('========================================')
    # print('Final top prediction is: ' + label_map[idx])
    # print('========================================')
    
    # print('========================================')
    # print('Inference time: ' + str(inference_time) + " ms")
    # print('========================================')
    
    # sort_idx = np.flip(np.squeeze(np.argsort(predictions)))
    # print('============ Top 5 labels are: ============================')
    # print(label_map[sort_idx[:5]])
    # print('===========================================================')
    