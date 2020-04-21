import tensorflow as tf
# https://github.com/tensorflow/tensorflow/issues/29124
if __name__ == "__main__":
    
    saved_model_dir = "./resnet_saved_model4"
    
    #converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir, tag_set=[tf.saved_model.tag_constants.SERVING])
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    #converter.allow_custom_ops = True
    #converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    
    tflite_model = converter.convert()
    open("converted_resnet.tflite", "wb").write(tflite_model)