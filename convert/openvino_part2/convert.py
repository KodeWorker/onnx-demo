# test a fig by the command : python .\predict.py --weights="ENetB0_4cls.h5"  --image="OK-N2-g.bmp"

import time
import numpy as np
# import pandas as pd
import tensorflow as tf
from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten , Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, losses
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from os import walk, listdir
from os.path import basename, dirname, isdir, isfile, join
import json
import argparse
from efficient.Im_prepro import *

#import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
import keras2onnx

class effnet:

    def __init__(self):

        self.imag_w, self.imag_h = 160, 160
        self.judge_result = {}
        self.class_names = {'0':'OK', '1':'NG-WL','2':'NG-BL','3':'NG-SP','4':'NG-LL'}
        #------------------------------------start building model-----------------------------# 
        model = EfficientNetB5(weights = None, input_shape = (self.imag_h,self.imag_w,3), include_top=False)

        ENet_out = model.output
        ENet_out = MaxPooling2D(pool_size=(2, 2))(ENet_out)
        ENet_out = Flatten()(ENet_out)

        Hidden1_in = Dense(1024, activation="relu")(ENet_out)
        Hidden1_in = Dropout(0.5)(Hidden1_in)

        predictions = Dense(units = 5, activation="softmax")(Hidden1_in) #3/12 改成預測5類
        self.model_f = Model(inputs = model.input, outputs = predictions)
        self.model_f.compile(optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=[metrics.mae, metrics.categorical_accuracy])
        self.model_f.load_weights("./efficient/ENetB5_5cls.h5") #3/12 改成預測5類

    #3/24 revise
    def predict_result(self, filename):
        self.judge_result = {} #每次執行預測要初始化
        for image_path in filename:

            img_prepro_time0 = time.time()

            imag_bgr_small, imag_c_small = preprocess_imgs(image_path) #3/24 revised
            tests = imag_c_small.reshape(-1,self.imag_h,self.imag_w,3)
            tests = tests.astype('float32') / 255
            
            pre_time1= time.time()

            test_predictions = self.model_f.predict(tests)
            y_test_pre = np.argmax(test_predictions,axis = 1)

            res_class = self.class_names[str(y_test_pre[0])] #3/24 added
            # print("the prob is {}".format(test_predictions)) #3/24 added
            print("the img name: {}; the predicted class: {}".format(basename(image_path),res_class)) #3/24 added
            
            pre_time2= time.time()
            
            if y_test_pre[0] > 0: #3/24改成只有一張圖判斷NG or OK
                self.judge_result.update({image_path:"NG"})
                print("{} is FAIL".format(basename(image_path)))
            else : 
                self.judge_result.update({image_path:"PASS"})
                print("{} is PASS".format(basename(image_path)))

            # print("just pre time cost =  %f s" % (pre_time2 - pre_time1))
            print("all time consumption = %f s\n" %(pre_time2 - img_prepro_time0))


        return self.judge_result

if __name__ == '__main__':

    net = effnet()
    
    #save_model_file = "panel.onnx"
    model = net.model_f
    print(model.inputs)
    #onnx_model = keras2onnx.convert_keras(model, model.name)
    #keras2onnx.save_model(onnx_model, save_model_file)