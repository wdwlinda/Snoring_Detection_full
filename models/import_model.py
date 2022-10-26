from pathlib import Path

import onnxruntime


class SavedModelBuilder():
    def __init__(self):
        pass

    def __call__(self, model_path):
        model_path = Path(model_path)
        if model_path.suffix == 'onnx':
            return self.from_onnx(model_path)
        elif model_path.suffix == 'tflite':
            return self.from_tflite(model_path)
        # TODO: Tensorflow model
        # TODO: Pytorch jit?
        # elif model_path.suffix == 'pb':

    def from_onnx(self, model_path):
        return onnxruntime.InferenceSession(model_path)

    def from_tflite(self, model_path, input_shape):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.resize_tensor_input(0, input_shape, strict=True)
        interpreter.allocate_tensors()
        return interpreter



def build_tflite(tflite_path, input_shape):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.resize_tensor_input(0, input_shape, strict=True)
    interpreter.allocate_tensors()
    return interpreter