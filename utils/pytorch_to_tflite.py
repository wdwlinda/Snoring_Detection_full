import os
import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
import tensorflow as tf
from onnx_tf.backend import prepare
# torch --> onnx


def pytorch_to_onnx(path, name, model, input_size):
    # set the model to inference mode 
    model.eval() 

    # Let's create a dummy input tensor  
    dummy_input = torch.randn(1, input_size, requires_grad=True)  

    # Export the model   
    torch.onnx.export(model,         # model being run 
         dummy_input,       # model input (or a tuple for multiple inputs) 
         os.path.join(path, f'{name}.onnx'),       # where to save the model  
         export_params=True,  # store the trained parameter weights inside the model file 
        #  opset_version=10,    # the ONNX version to export the model to 
        #  do_constant_folding=True,  # whether to execute constant folding for optimization 
         input_names = ['modelInput'],   # the model's input names 
         output_names = ['modelOutput'], # the model's output names 
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes 
                       'modelOutput' : {0 : 'batch_size'}}) 
    print(" ") 
    print('Model has been converted to ONNX') 
    # model = onnx.load("mobilenet_v2.onnx")
    # ort_session = ort.InferenceSession('mobilenet_v2.onnx')
    # onnx_outputs = ort_session.run(None, {'input': test_arr})


def onnx_to_tf(path, name):
    tf_path = os.path.join(path, name)
    onnx_path = os.path.join(path, f'{name}.onnx')
    onnx_model = onnx.load(onnx_path)  # load onnx model
    tf_rep = prepare(onnx_model)  # creating TensorflowRep object
    tf_rep.export_graph(tf_path)


def tf_to_tflite(path, name):
    tf_path = os.path.join(path, name)
    tflite_path = os.path.join(path, f'{name}.tflite')
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tf_lite_model = converter.convert()
    with open(tflite_path, 'wb') as f:
        f.write(tf_lite_model)


def convert_pytorch_to_tf_lite(path, model_name):
    model = Network() 
    path = "myFirstModel.pth" 
    model.load_state_dict(torch.load(path)) 
    input_size = 0
    pytorch_to_onnx(path, model_name, model, input_size)
    onnx_to_tf(path, model_name)
    tf_to_tflite(path, model_name)


def main():
    convert_pytorch_to_tf_lite()


if __name__ == '__main__':
    main()