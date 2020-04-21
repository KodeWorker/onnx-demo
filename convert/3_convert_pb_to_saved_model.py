# -*- coding: utf-8 -*-
import onnx
from onnx_tf.backend import prepare
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

if __name__ == "__main__":
        
    onnx_file_path = r"./opt-efficientnet-b1_rename.onnx"
    frozen_graph_pb = r"./opt-efficientnet-b1_toco.pb"
    export_dir = r"./opt-efficientnet-b1_saved_model"
    
    onnx_model = onnx.load(onnx_file_path)  # load onnx model
    # convert tf *.pb model to saved model
    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
    input_name = onnx_model.graph.node[0].input[0] + ":0"
    output_name = onnx_model.graph.node[-1].output[0] + ":0"
    
    print("input node name: {}".format(input_name))
    print("output node name: {}".format(output_name))
    
    with tf.gfile.GFile(frozen_graph_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())    
    
    sigs = {}
    
    with tf.Session(graph=tf.Graph()) as sess:
        # name="" is important to ensure we don't get spurious prefixing
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)
    
        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})
    
        builder.add_meta_graph_and_variables(sess,
                                              [tag_constants.SERVING],
                                              signature_def_map=sigs)
    
    builder.save()