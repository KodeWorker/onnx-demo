# -*- coding: utf-8 -*-
import onnx
from onnx_tf.backend import prepare
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

if __name__ == "__main__":
    
    # onnx_file_path = r"../opt-efficientnet-b1.onnx"
    # frozen_graph_pb = r"opt-efficientnet-b1.pb"
    # export_dir = r"./efficientnet_saved_model"
    
    onnx_file_path = r"../resnet50v2.onnx"
    frozen_graph_pb = r"resnet50v2.pb"
    export_dir = r"./resnet_saved_model"
    
    # convert onnx to tf *.pb model
    onnx_model = onnx.load(onnx_file_path)  # load onnx model
    
    ##############################################
    # https://github.com/onnx/onnx-tensorflow/issues/589
    for init_vals in onnx_model.graph.initializer:
    	init_vals.name = 'tf_' + init_vals.name

    for inp in onnx_model.graph.input:
    	inp.name = 'tf_' + inp.name
    
    for op in onnx_model.graph.output:
    	op.name = 'tf_' + op.name
    
    for node in onnx_model.graph.node:
    	node.name = 'tf_' + node.name
    	for i in range(len(node.input)):
    		node.input[i] = 'tf_' + node.input[i]
    	for i in range(len(node.output)):
    		node.output[i] = 'tf_' + node.output[i]	
    ##############################################
    
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(frozen_graph_pb)  # export the model
    
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