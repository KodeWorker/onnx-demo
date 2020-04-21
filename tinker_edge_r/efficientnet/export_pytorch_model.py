# -*- coding: utf-8 -*-
import torch 
from efficientnet_pytorch import EfficientNet

if __name__ == "__main__":
    
    onnx_file = "efficientnet-b1.onnx"
    
    model = EfficientNet.from_pretrained('efficientnet-b1')    
    model.set_swish(memory_efficient=False)
    model.eval()
    
    example_input = torch.Tensor(1, 3, 224, 224) # 224 is the least input size, depends on the dataset you use

    script_module = torch.jit.trace(model, example_input)
    script_module.save('script_module.pt')
    
    
    #load_model = torch.jit.load('script_module.pt')