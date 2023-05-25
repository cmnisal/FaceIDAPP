# import onnx
# import torch
# import numpy as np
# model = torch.load("FaceTransformerOctupletLoss.pt")

# # TODO Try on UBUNTU MACHINE NOT WORKING ON MAC

# # print(torch.backends.quantized.supported_engines)

# # model.qconfig = torch.quantization.get_default_qconfig('qnnpack')



# # torch.quantization.prepare(model, inplace=True)
# # model.eval()
# # torch.quantization.convert(model, inplace=True)

# input_names = ["input_image"]
# output_names = ["output"]
# dummy_input = torch.randn(1, 3, 112, 112)

# torch.onnx.export(model,
#                  dummy_input,
#                  "FaceTransformerOctupletLoss.onnx",
#                  verbose=True,
#                  input_names=input_names,
#                  output_names=output_names,
#                  export_params=True,
#                  opset_version=13,
#                  dynamic_axes={'input_image': {0: 'batch_size'},
#                                'output': {0: 'batch_size'}
#                                }
#                  )

# TESTING
# import numpy as np
# import onnxruntime as onnxrt
# onnx_session= onnxrt.InferenceSession("FaceTransformerOctupletLoss.onnx", providers=onnxrt.get_available_providers())
# onnx_inputs= {onnx_session.get_inputs()[0].name:
# np.random.rand(1, 3, 112, 112).astype(np.float32)}
# onnx_output = onnx_session.run(None, onnx_inputs)
# img_label = onnx_output[0]

# Make Model Outputs and Inputs Batch Size None: of ONNX model
# model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = '?'
# model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
