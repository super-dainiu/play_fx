# play_fx
Playing torch.FX for multiple usages

## FX Graph
See https://pytorch.org/docs/stable/fx.html. This is clear enough.

## Orig FX passes
```python
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
N, D_in, H, D_out = 64, 1000, 100, 10
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
model = TwoLayerNet(D_in, H, D_out)
gm = torch.fx.symbolic_trace(model)
sample_input = torch.randn(50, D_in)
ShapeProp(gm).propagate(sample_input)
for node in gm.graph.nodes:
    print(node.name, node.meta['tensor_meta'].dtype,
        node.meta['tensor_meta'].shape) 
```
