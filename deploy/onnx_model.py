import torch
# import torch.onnx
import numpy as np
import onnxruntime


# TODO: wrap to class
def torch_to_onnx(
        dummy_input, model, save_filename, input_names=['input'], output_names=['output'],
        dynamic_axes={'input' : {1: 'channel', 2 : 'height', 3: 'width'},
                      'output' : {1: 'num_class', 2: 'height', 3: 'width'}}
    ):
    model = model.eval()
    
    # TODO: assert for input, output, dynamic_axes
    if isinstance(dummy_input, (list, tuple)): 
        dummy_input_t = []
        for np_input in dummy_input:
            if isinstance(np_input, (float, int)):
                inputs = torch.tensor(np_input)
            else:
                inputs = torch.from_numpy(np_input)
            # inputs = inputs.cuda()
            dummy_input_t.append(inputs)
        dummy_input_t = tuple(dummy_input_t)
    else:
         dummy_input_t = torch.from_numpy(dummy_input)
         dummy_input = [dummy_input]
        #  dummy_input_t = dummy_input_t.cuda()
    
    # torch inference
    with torch.no_grad():
        if isinstance(dummy_input_t, (list, tuple)):
            torch_out_t = model(*dummy_input_t)
        else:
            torch_out_t = model(dummy_input_t)

        if isinstance(torch_out_t, (list, tuple)):
            torch_out = []
            for tensor in list(torch_out_t):
                torch_out.append(to_numpy(tensor))
        else:
            torch_out = [to_numpy(torch_out_t)]

    # ONNX model exporting
    torch.onnx.export(
        model,               # model being run
        dummy_input_t,                         # model input (or a tuple for multiple inputs)
        save_filename,   # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=input_names,   # the model's input names
        output_names=output_names, # the model's output names
        dynamic_axes=dynamic_axes
    )

    # ONNX inference
    ort_outs = ONNX_inference(dummy_input, save_filename)

    # compare ONNX Runtime and PyTorch results
    for t, o in zip(torch_out, ort_outs):
        error = t - o
        
        print(error.max(), error.min(), np.abs(error).sum())

        np.testing.assert_allclose(t, o, rtol=1e-03, atol=1e-5)
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def ONNX_inference(inputs, onnx_model):
    """AI is creating summary for ONNX_inference

    Args:
        inputs ([type]): [description]
        onnx_model ([type]): [description]

    Returns:
        [type]: [description]
    """
    ort_session = onnxruntime.InferenceSession(onnx_model)
    # compute ONNX Runtime output prediction
    input_names = ort_session.get_inputs()
    assert len(inputs) == len(input_names)
    ort_inputs = {
        input_session.name: input_data for input_session, input_data in zip(input_names, inputs)}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs