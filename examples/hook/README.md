# VNE regularization using hook

The regularization of von Neumann entropy (VNE) in representation learning tasks can be easily incorporated into existing models with minimal modification to the training code. By employing this technique, we can effectively balance the objectives of representation learning and regularization, resulting in high-quality and stable representations.
This can be achieved using a **hook** in the PyTorch framework, which allows for the retrieval of intermediate layer representations during training process.
To illustrate this approach, we present an example that demonstrates the followings:
1. the retrieval of intermediate layer representations using hook.
2. the incorporation of VNE regularization into the main loss via the subtraction of the VNE from the main loss value.



#### The following code example demonstrates how to regularize VNE of fc_layer representation of ResNet-18 model when training supervised tasks.


###### 0. Register forward hook to the fc_layer of ResNet-18 (before training)
```py
import torchvision
model = torchvision.models.resnet18()

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        self.input = input[0]
    def close(self):
        self.hook.remove()

hook_fc = Hook(model.fc)
```

###### 1. Retrieve 'x', the fc_layer representation corresponding to current batch of data 'data'

```py
outputs = model(data.to('cuda'))

x = hook_fc.input
```

###### 2. Incorporate VNE with the main cross-entropy loss

```py
loss = torch.nn.functional.cross_entropy(outputs, targets.to('cuda'))

if abs(vne_coef) > 0:
    loss -= vne_coef * get_vne(x)

```




