# -*- coding: utf-8 -*-
import onnx
from onnx_tf.backend import prepare

if __name__ == "__main__":
    
    onnx_file_path = r"../opt-efficientnet-b1.onnx"
    export_file_path = r"./opt-efficientnet-b1_rename.onnx"
    frozen_graph_pb = "opt-efficientnet-b1.pb"
    
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
    
    onnx.save(onnx_model, export_file_path)
    
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph(frozen_graph_pb)  # export the model