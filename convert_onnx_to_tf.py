import onnx
from onnx_tf.backend import prepare

if __name__ == "__main__":
    onnx_model = onnx.load("mobilenetv2-1.0.onnx")  # load onnx model
    tf_rep = prepare(onnx_model)  # prepare tf representation
    tf_rep.export_graph("mobilenetv2-1.0-tf.pb")  # export the model