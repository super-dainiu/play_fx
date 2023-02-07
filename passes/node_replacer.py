import torch
import torch.nn as nn
import torch.fx as fx
from typing import Any, Dict, List, Tuple


class ConvChannelReduction(fx.Interpreter):
    """
    Reduce the number of channels in a Conv2d layer by a factor of reduction_factor.
    """
    def __init__(self, gm, reduction_factor=0.5):
        super().__init__(gm)
        self.reduction_factor = reduction_factor
    
    def nop(self, *args, **kwargs):
        pass
    
    call_function = nop
    call_method = nop
    get_attr = nop
    placeholder = nop
    output = nop

    def run_node(self, n : fx.Node) -> Any:
        """
        Run a specific node ``n`` and return the result.
        Calls into placeholder, get_attr, call_function,
        call_method, call_module, or output depending
        on ``node.op``

        Args:
            n (Node): The Node to execute

        Returns:
            Any: The result of executing ``n``
        """
        args, kwargs = self.fetch_args_kwargs_from_env(n)
        assert isinstance(args, tuple)
        assert isinstance(kwargs, dict)
        return getattr(self, n.op)(n.target, args, kwargs)

    
    def call_module(self, target : str, args : Tuple[Any, ...], kwargs : Dict[str, Any]) -> Any:
        # Execute the method and return the result
        assert isinstance(target, str)
        target_atoms = target.split('.')
        attr_itr = self.module
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
            par_itr = attr_itr
            attr_itr = getattr(attr_itr, atom)
        
        if isinstance(attr_itr, nn.modules.conv._ConvNd):
            repl = torch.nn.Conv2d(
                in_channels=int(attr_itr.in_channels * self.reduction_factor),
                out_channels=int(attr_itr.out_channels * self.reduction_factor),
                kernel_size=attr_itr.kernel_size,
                stride=attr_itr.stride,
                padding=attr_itr.padding,
                dilation=attr_itr.dilation,
                groups=attr_itr.groups,
                bias=attr_itr.bias is not None,
                padding_mode=attr_itr.padding_mode
                )
            delattr(par_itr, target_atoms[-1])
            setattr(par_itr, target_atoms[-1], repl)


def compute_size_in_bytes(elem: torch.Tensor | Dict | List | Tuple | int) -> int:
    """Compute the size of a tensor or a collection of tensors in bytes.

    Args:
        elem (torch.Tensor | Dict | List | Tuple | int): Arbitrary nested ``torch.Tensor`` data structure.

    Returns:
        int: The size of the tensor or the collection of tensors in bytes.
    """
    nbytes = 0
    if isinstance(elem, torch.Tensor):
        if elem.is_quantized:
            nbytes += elem.numel() * torch._empty_affine_quantized([], dtype=elem.dtype).element_size()
        else:
            nbytes += elem.numel() * torch.tensor([], dtype=elem.dtype).element_size()
    elif isinstance(elem, dict):
        value_list = [v for _, v in elem.items()]
        nbytes += compute_size_in_bytes(value_list)
    elif isinstance(elem, tuple) or isinstance(elem, list) or isinstance(elem, set):
        for e in elem:
            nbytes += compute_size_in_bytes(e)
    return nbytes


if __name__ == "__main__":
    import torchvision.models as tm
    from torch.autograd.profiler_util import _format_memory
    
    m = tm.resnet18()
    gm = fx.symbolic_trace(m)
    
    orig_param_size = compute_size_in_bytes(tuple(gm.parameters()))
    print(f"Original model parameter size: {_format_memory(orig_param_size)}")

    replacer = ConvChannelReduction(gm)
    replacer.run()

    new_param_size = compute_size_in_bytes(tuple(gm.parameters()))
    print(f"New model parameter size: {_format_memory(new_param_size)}")

