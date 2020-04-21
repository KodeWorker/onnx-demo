import onnx
from onnx import optimizer

if __name__ == "__main__":
    source = "efficientnet-b1.onnx"
    target = "opt-efficientnet-b1.onnx"
    
    onnx_model = onnx.load(source)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)
    polish_model = onnx.utils.polish_model(optimized_model)
    onnx.save(polish_model, target)
    
    # onnx.checker.check_model(onnx_model)
    #onnx.helper.printable_graph(onnx_model.graph)