import efficientnet.tfkeras as efn
import keras2onnx

if __name__ == "__main__":
    temp_model_file = 'efficientnet-b1.onnx'

    model = efn.EfficientNetB0(weights='imagenet')
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, temp_model_file)