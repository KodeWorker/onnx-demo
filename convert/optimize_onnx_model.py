import onnx
from onnx import optimizer

if __name__ == "__main__":
    source = "efficientnet-b7.onnx"
    target = "opt-" + source
    
    onnx_model = onnx.load(source)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)
    onnx.save(optimized_model, target)
    
    #onnx.checker.check_model(onnx_model)
    #onnx.helper.printable_graph(onnx_model.graph)