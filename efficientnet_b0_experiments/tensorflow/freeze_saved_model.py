#import tensorflow as tf
from tensorflow.python.tools import freeze_graph
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
#import tf2onnx

def load_pb(path_to_pb):
    with tf.io.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
    
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == "__main__":

    """
    export_dir = "./output_model"
    loaded = tf.saved_model.load(export_dir)
    infer = loaded.signatures["serving_default"]
    print(infer)
    
    file="dog.jpg"
    img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = tf.keras.applications.mobilenet.preprocess_input(
        x[tf.newaxis,...])
    """
    
    
    export_dir = "./output_model"
    path_to_pb = "frozen_model.pb"
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_dir)
        graph = tf.get_default_graph()
        
        #x = sess.graph.get_tensor_by_name('images:0')
        #y = sess.graph.get_tensor_by_name('Softmax:0')
        
        #onnx_graph = tf2onnx.tfonnx.process_tf_graph(sess.graph, input_names=['images:0'], output_names=['Softmax:0'])
        
        od_graph_def = convert_variables_to_constants(sess, graph.as_graph_def(), ['Softmax'])
        with tf.io.gfile.GFile(path_to_pb, "wb") as f:
            f.write(od_graph_def.SerializeToString())
        
    """
    path_to_pb="./output_model/saved_model.pb"
    grapg = load_pb(path_to_pb)
    """