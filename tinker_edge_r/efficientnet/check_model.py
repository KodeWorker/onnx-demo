# -*- coding: utf-8 -*-

import onnx

if __name__ == "__main__":
    
    ONNX_MODEL = 'opt-efficientnet-b1.onnx'
    onnx_model = onnx.load(ONNX_MODEL)
    
    polish_model = onnx.utils.polish_model(onnx_model)