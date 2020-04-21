import onnx
import tensorflow.compat.v1 as tf
# https://github.com/tensorflow/tensorflow/issues/29124
# https://stackoverflow.com/questions/53182177/how-do-you-convert-a-onnx-to-tflite
if __name__ == "__main__":
        
    onnx_file_path = r"./resnet50v2_rename.onnx"
    frozen_graph_pb = r"./resnet50v2_toco.pb"
    
    #converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, tag_set=[tf.saved_model.tag_constants.SERVING])
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    onnx_model = onnx.load(onnx_file_path)  # load onnx model
    input_name = onnx_model.graph.node[0].input[0]
    output_name = onnx_model.graph.node[-1].output[0]
    
    converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_graph_pb, input_arrays=[input_name], output_arrays=[output_name])
    # tell converter which type of optimization techniques to use
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    open("converted_resnet.tflite", "wb").write(tflite_model)