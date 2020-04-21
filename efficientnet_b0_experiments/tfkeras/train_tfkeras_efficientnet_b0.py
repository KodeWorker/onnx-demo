# -*- coding: utf-8 -*-
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.compat.v1 import set_random_seed
from tensorflow import keras
import os
from PIL import Image
import numpy as np
from tensorflow.keras.utils import to_categorical

class tfkeras_dataloader:
    def __init__(self, data_dir, img_size):
        self.data_dir = data_dir
        self.img_size = img_size

    def generate(self, mean, std):
        x = []
        y = []
        
        dirs = os.listdir(self.data_dir)
        for i in range(len(dirs)):
            filenames = os.listdir(os.path.join(self.data_dir, dirs[i]))
            for filename in filenames:
                filepath = os.path.join(self.data_dir, dirs[i], filename)
                image = Image.open(filepath)
                image = image.resize((self.img_size, self.img_size))
                img_array = np.array(image)
                for n_channel in range(img_array.shape[-1]):
                    img_array[:, :, n_channel] = (img_array[:, :, n_channel]/255 - mean[n_channel]) / std[n_channel]
                x.append(img_array)
                y.append(i)
        x, y = np.asarray(x), np.asarray(y)
        y = to_categorical(y)
        return x, y

if __name__ == "__main__":
    
    RANDOM_SEED = 5566
    NUM_CLASSES = 2
    EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    IMG_SIZE = 224
    
    set_random_seed(RANDOM_SEED)
    model = EfficientNetB0(weights=None, classes=NUM_CLASSES)
    #print(model.summary())
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),  # Optimizer
                  # Loss function to minimize
                  loss=keras.losses.categorical_crossentropy,
                  # List of metrics to monitor
                  metrics=[keras.metrics.categorical_accuracy])
    
    datadir = r"D:\Datasets\NB-CONN\divided"
    traindir = os.path.join(datadir, "train")
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    data_loader = tfkeras_dataloader(traindir, img_size=IMG_SIZE)
    x_train, y_train = data_loader.generate(mean, std)
    
    print('# Fit model on training data')
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS)    
    print('\nhistory dict:', history.history)
    
    model_path = 'tfkeras_efficientnet_b0_weights.h5'
    model.save_weights(model_path)