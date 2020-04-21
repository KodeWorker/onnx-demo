import os
import torch 
from efficientnet_pytorch import EfficientNet
import onnx
from onnx import optimizer

if __name__ == "__main__":
    
    onnx_file = "efficientnet-b1.onnx"
    temp_file = "efficientnet-b1.temp"
    
    model = EfficientNet.from_pretrained('efficientnet-b1')    
    model.set_swish(memory_efficient=False) #!!!
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    input_names = [ "input" ]
    output_names = [ "output" ]
    
    torch.onnx.export(model, dummy_input, temp_file, verbose=True, input_names=input_names, output_names=output_names,
    keep_initializers_as_inputs=True)
    
    source = temp_file
    target = onnx_file
    
    onnx_model = onnx.load(source)
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
    optimized_model = optimizer.optimize(onnx_model, passes)
    onnx.save(optimized_model, target)
    os.remove(temp_file)