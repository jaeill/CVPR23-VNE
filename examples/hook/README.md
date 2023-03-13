# VNE regularization using hook

The following code example demonstrates how to regularize VNE of fc_layer representation of ResNet-18 model when training supervised tasks.

```py
import torch
import torchvision
from vne import get_vne

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input[0]
    def close(self):
        self.hook.remove()

model = torchvision.models.resnet18()

hook_fc = Hook(model.fc)

data, targets = next(iter(loader_train))

outputs = model(data.to('cuda'))

loss = torch.nn.functional.cross_entropy(outputs, targets.to('cuda'))

x = hook_fc.input

if abs(vne_coef) > 0:
    loss -= vne_coef * get_vne(x)


```




