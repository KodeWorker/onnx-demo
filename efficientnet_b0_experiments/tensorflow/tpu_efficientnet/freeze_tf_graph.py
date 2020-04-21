from tensorflow.python.tools import freeze_graph 
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.python.framework import graph_util

if __name__ == "__main__":
    model_folder = "../EfficentNet-b0"
    output_filename = "frozen-graph.pb"
    output_nodes = "output"
    rename_outputs = None
    
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_graph = output_filename
    
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', 
                                       clear_devices=True)
    graph = tf.get_default_graph()
    onames = output_nodes.split(',')
    
    if rename_outputs is not None:
        nnames = rename_outputs.split(',')
        with graph.as_default():
            for o, n in zip(onames, nnames):
                _out = tf.identity(graph.get_tensor_by_name(o+':0'), name=n)
            onames=nnames
    
    input_graph_def = graph.as_graph_def()
    
    # fix batch norm nodes
    for node in input_graph_def.node:
        if node.op == 'RefSwitch':
            node.op = 'Switch'
            for index in range(len(node.input)):
                if 'moving_' in node.input[index]:
                    node.input[index] = node.input[index] + '/read'
        elif node.op == 'AssignSub':
            node.op = 'Sub'
            if 'use_locking' in node.attr: del node.attr['use_locking']
    
    # with tf.Session(graph=graph) as sess:
    #     saver.restore(sess, input_checkpoint)

    #     # In production, graph weights no longer need to be updated
    #     # graph_util provides utility to change all variables to constants
    #     output_graph_def = graph_util.convert_variables_to_constants(
    #         sess, input_graph_def, 
    #         onames # unrelated nodes will be discarded
    #     ) 

    #     # Serialize and write to file
    #     with tf.gfile.GFile(output_graph, "wb") as f:
    #         f.write(output_graph_def.SerializeToString())
    #     print("%d ops in the final graph." % len(output_graph_def.node))