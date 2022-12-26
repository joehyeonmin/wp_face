import torch
import torchvision
from torch.utils.mobile_optimizer import optimize_for_mobile

# num_classes -> embedding size?
model = torchvision.models.mobilenet_v3_small(num_classes=512)
model.eval()
example = torch.rand(1, 3, 224, 224)
loaded_state = torch.load("model0100.model", map_location=torch.device('cpu'))
self_state = model.state_dict()

# print(self_state.items())
    
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

# torch.save(model, "model.pt")  
    
traced_script_module = torch.jit.script(model)
traced_script_module_optimized = optimize_for_mobile(traced_script_module)
traced_script_module_optimized._save_for_lite_interpreter("jit_model.pt")