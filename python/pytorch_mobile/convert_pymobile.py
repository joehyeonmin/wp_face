import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile
import ssl

from torch.jit.mobile import (
    _backport_for_mobile,
    _get_model_bytecode_version,
)

ssl._create_default_https_context = ssl._create_unverified_context
print(torch.__version__)
print(torchvision.__version__)

model = torchvision.models.mobilenet_v3_small                                                                                                                                                                                                                                                       (pretrained=True)
model.eval()

mobilenet_v3 = torch.quantization.convert(model)

example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(mobilenet_v3, example)
optimized_traced_model = optimize_for_mobile(traced_script_module)
optimized_traced_model._save_for_lite_interpreter("mobilenet_V3.pt")

MODEL_INPUT_FILE = "mobilenet_V3.pt"
MODEL_OUTPUT_FILE = "mobilenet_V4.pt"

print("model version : ", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

_backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=5)

# print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE)._save_for_lite_interprete)
print("new model version : ", _get_model_bytecode_version(MODEL_OUTPUT_FILE))