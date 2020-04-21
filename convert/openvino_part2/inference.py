from os.path import basename, join
from os import listdir
import numpy as np
from efficient.Im_prepro import preprocess_imgs
import time 
import onnxruntime

def image_unit_test(image, onnx_sess):
    #test_predictions = self.model_f.predict(tests)
    t0 = time.time()
    test_predictions = onnx_sess.run([output_name], {input_name: image})[0]
    t = time.time() - t0
    #print(test_predictions)
    y_test_pre = np.argmax(test_predictions,axis = 1)
    res_class = class_names[str(y_test_pre[0])] #3/24 added
    print("the img name: {}; the predicted class: {}".format(basename(image_path),res_class)) #3/24 added
    
    return res_class, t

if __name__ == "__main__":
    
    onnx_model_path = "panel.onnx"
    class_names = {'0':'OK', '1':'NG-WL','2':'NG-BL','3':'NG-SP','4':'NG-LL'}
    
    sess = onnxruntime.InferenceSession(onnx_model_path)
    #sess.set_providers(['CPUExecutionProvider'])
    
    # input
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    # output
    output_name = sess.get_outputs()[0].name   
    
    #image_dir = "efficient/data/0-OK"
    #image_dir = "efficient/data/1-NG-WL"
    #image_dir = "efficient/data/2-NG-BL"
    image_dir = "efficient/data/3-NG-Spot"
    #image_dir = "efficient/data/4-NG-LL"
    
    # data/3-NG-Spot/fine_line_k.bmp 9.8998308e-01 4.3665328e-05 8.0722106e-07 9.9322619e-03 4.0115519e-05
    # Kera results: 9.8998308e-01 4.3664666e-05 8.0720025e-07 9.9322908e-03 4.0114639e-05 
    
    total_time = 0
    for filename in listdir(image_dir):        
        image_path = join(image_dir, filename)
        imag_bgr_small, imag_c_small = preprocess_imgs(image_path) #3/24 revised
        tests = imag_c_small.reshape(-1,input_shape[1],input_shape[2],3)
        tests = tests.astype('float32') / 255
        res_class, t = image_unit_test(tests, sess)
        total_time += t
    print("Time Elapsed: {:.4f} sec./pic".format(total_time/len(listdir(image_dir))))
       