import torch 
from efficientnet_pytorch import EfficientNet

if __name__ == "__main__":
    
    onnx_file = "efficientnet-b1.onnx"
    
    model = EfficientNet.from_pretrained('efficientnet-b1')    
    model.set_swish(memory_efficient=False) #!!!
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    input_names = [ "input" ]
    output_names = [ "output" ]
    
    torch.onnx.export(model, dummy_input, onnx_file, verbose=True, input_names=input_names, output_names=output_names,
    keep_initializers_as_inputs=True)
    