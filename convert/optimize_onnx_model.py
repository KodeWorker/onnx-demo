import onnx
from onnx import optimizer

if __name__ == "__main__":
    
    for model_name in ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7']:
        
        source = model_name + ".onnx"
        target = "opt-" + source
        
        onnx_model = onnx.load(source)
        passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
        optimized_model = optimizer.optimize(onnx_model, passes)
        onnx.save(optimized_model, target)
        
        #onnx.checker.check_model(onnx_model)
        #onnx.helper.printable_graph(onnx_model.graph)