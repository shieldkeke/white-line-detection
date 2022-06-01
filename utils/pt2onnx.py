import torch
from torch.autograd import Variable
import torch.onnx as torch_onnx
from model import MobileNetV3_Small

input_shape = (3, 640, 480)
model_onnx_path = "torch_model.onnx"
model = MobileNetV3_Small()
model.train(False)

# Export the model to an ONNX file
##### ONLY FOR DISPLAY!######
dummy_input = Variable(torch.randn(1, *input_shape))
output = torch_onnx.export(model, 
                          dummy_input, 
                          model_onnx_path, 
                          verbose=False)
print("Export of torch_model.onnx complete!")