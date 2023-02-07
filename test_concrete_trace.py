from nni.common.concrete_trace_utils import concrete_trace
import torch
from zoo import tm_models, tmm_models

def test_concrete_trace(mod, input, use_function_patch=True):
    model = mod()
    model.eval()
    kwargs = dict(x=input)
    gm = concrete_trace(model, concrete_args=kwargs, use_function_patch=use_function_patch)
    
    assert torch.allclose(model(input), model(input)), f"concrete trace result of {mod.__name__} is not correct"
    with torch.no_grad():
        assert torch.allclose(model(input), gm(input)), f"concrete trace result of {mod.__name__} is not correct"

def test_concrete_trace_torchvision():
    for mod in tm_models:
        test_concrete_trace(mod, torch.rand(8, 3, 224, 224))
    
def test_concrete_trace_timm():
    for mod in tmm_models:
        test_concrete_trace(mod, torch.rand(1, 3, 224, 224))

if __name__ == '__main__':
    test_concrete_trace_torchvision()
    test_concrete_trace_timm()
