import torch
import torchvision

from torch.utils.mobile_optimizer import optimize_for_mobile

model = torchvision.models.mobilenet_v3_small(num_classes=512)
model.eval()

example = torch.rand(1, 3, 224, 224)
loaded_state = torch.load("model0300.model", map_location=torch.device('cpu'))
self_state = model.state_dict()

for name, param in loaded_state.items():     
    #print(name)
    
    origname = name[6:]
    if origname not in self_state:
        if origname not in self_state:
            print("{} is not in the model.".format(origname))
            continue
    if self_state[origname].size() != loaded_state[name].size():
        print("Wrong parameter length: {}, model: {}, loaded: {}".format(origname, self_state[origname].size(), loaded_state[name].size()));
        continue

    model.state_dict()[origname].copy_(param)   
    # (model.state_dict())[origname].copy_(param)   

    
traced_script_module = torch.jit.trace(model, example)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("_test_jit_model.ptl")


# from torch.jit.mobile import (
#     _backport_for_mobile,
#     _get_model_bytecode_version,
# )

# MODEL_INPUT_FILE = "jit_model.pt"
# MODEL_OUTPUT_FILE = "down_mobilenet_V3_small.pt"

# print("model version : ", _get_model_bytecode_version(f_input=MODEL_INPUT_FILE))

# _backport_for_mobile(f_input=MODEL_INPUT_FILE, f_output=MODEL_OUTPUT_FILE, to_version=5)

# # print("new model version", _get_model_bytecode_version(MODEL_OUTPUT_FILE)._save_for_lite_interprete)
# print("new model version : ", _get_model_bytecode_version(MODEL_OUTPUT_FILE))