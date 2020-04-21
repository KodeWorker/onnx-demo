import tflite_runtime.interpreter as tflite
import numpy as np
from PIL import Image

def load_labels(label_path):
    with open(label_path, "r") as f:
        return [line.strip() for line in f.readlines()]

def preprocess(input_image, mean_vec, stddev_vec):
    output_image = np.zeros(input_image.shape, dtype=np.float32)
    for i in range(input_image.shape[-1]):
        output_image[:,:,i] = (input_image[:,:,i]/255. - mean_vec[i]) / stddev_vec[i]
    output_image = np.expand_dims(output_image, axis=0)
    return output_image

if __name__ == "__main__":
    
    n_top = 5
    mean_vec = [0.485, 0.456, 0.406]
    stddev_vec = [0.229, 0.224, 0.225]
    image_path = "./cat.1.jpg"
    model_path = "./pretrained_efficientnet_b0.tflite"
    label_path = "./tflite_label_map.txt"

    interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")])
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
     
    is_floating_model = input_details[0]["dtype"] == np.float32
    print(is_floating_model)

    height = input_details[0]["shape"][1]
    width = input_details[0]["shape"][2]

    input_image = Image.open(image_path).convert("RGB").resize((height, width))
    input_image = np.array(input_image, dtype=np.float32)
    input_image = preprocess(input_image, mean_vec, stddev_vec)

    print("input shape: {}".format(input_image.shape))
    interpreter.set_tensor(input_details[0]["index"], input_image)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])
    results = np.squeeze(output_data)
    print("output shape: {}".format(results.shape))

    top_k = results.argsort()[-n_top:][::-1]
    labels = load_labels(label_path)
    
    for i in top_k:
        print("{:08.6f}: {}".format(float(results[i]), labels[i]))
