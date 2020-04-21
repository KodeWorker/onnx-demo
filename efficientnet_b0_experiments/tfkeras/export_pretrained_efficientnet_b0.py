import efficientnet.tfkeras as efn

# https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.keras import backend as K
tf.disable_eager_execution()

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

if __name__ == "__main__":
    K.set_learning_phase(0)
    model = efn.EfficientNetB0(weights='imagenet')
    model._make_predict_function()
    #y = model.predict(np.random.rand(1, 224, 224, 3))
    
    # 2.1 Save Keras Model as TF2.0 SavedModel
    # save_dir = "./saved_model"
    # if os.path.exists(save_dir):
    #     shutil.rmtree(save_dir)
    # os.makedirs(save_dir)
    # model.save(save_dir, save_format='tf')
    
    # 2.2 Save Keras Model as FrozenGraph
    frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
    tf.train.write_graph(frozen_graph, "export", "frozen_model.pb", as_text=False)