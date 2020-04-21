import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import keras2onnx
# import onnxruntime

# image preprocessing
img_path = 'dog.jpg'   # make sure the image is in img_path
img_size = 224
img = image.load_img(img_path, target_size=(img_size, img_size))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# load keras model
from tensorflow.keras.applications.resnet50 import ResNet50
model = ResNet50(include_top=True, weights='imagenet')

# convert to onnx model
onnx_model = keras2onnx.convert_keras(model, model.name)

temp_model_file = 'model.onnx'
keras2onnx.save_model(onnx_model, temp_model_file)
# sess = onnxruntime.InferenceSession(temp_model_file)

# # runtime prediction
# content = onnx_model.SerializeToString()
# sess = onnxruntime.InferenceSession(content)
# x = x if isinstance(x, list) else [x]
# feed = dict([(input.name, x[n]) for n, input in enumerate(sess.get_inputs())])
# pred_onnx = sess.run(None, feed)