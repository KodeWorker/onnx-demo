import os
import time
import numpy as np
import tensorflow as tf

img_size = 360

###CPU

with tf.device('/CPU:0'):
    start_time = time.time()
    
    a = np.random.rand(img_size, img_size)
    var1 = tf.Variable(a)
    for _ in range(1000):
        var1 = tf.math.add(var1, var1)
        
    elapsed_time = time.time() - start_time
    print('CPU time = ',elapsed_time)

###GPU
with tf.device('/GPU:0'):
    start_time = time.time()
    
    b = np.random.rand(img_size, img_size)
    var2 = tf.Variable(b)
    for _ in range(1000):
        var2 = tf.math.add(var2, var2)
    
    elapsed_time = time.time() - start_time
    print('GPU time = ',elapsed_time)