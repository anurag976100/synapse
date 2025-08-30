import numpy as np
from typing import Callable
from enum import Enum
from synapse.autodiff.node import SynapseBackwardPropNode
from synapse.autodiff.context import SynapseComputeContext
from synapse.tensor.tensor_ops import Add


class SynapseDtype(Enum):
    float32 = 0
    float64 = 1
    int32 = 2
    int64 = 3


class SynapseTensor:
    def __init__(
        self,
        data: list | tuple | np.ndarray,
        *,
        shape: list[int] | tuple[int],
        dtype: SynapseDtype | np.dtype,
        requires_grad: bool = False,
        grad_fn: SynapseBackwardPropNode | None = None,
    ):
        assert (
            isinstance(dtype, np.dtype)
            or _SYNAPSE_DTYPE_TO_NP_DTYPE_MAP.get(dtype) is not None
        ), "Could not find requested tensor dtype in supported dtype mappings"
        self.data: np.ndarray = (
            data
            if isinstance(data, np.ndarray)
            else np.array(
                data,
                dtype=(
                    _SYNAPSE_DTYPE_TO_NP_DTYPE_MAP[dtype]
                    if not isinstance(dtype, np.dtype)
                    else dtype
                ),
            ).reshape(shape)
        )
        self.requires_grad: bool = requires_grad
        self.dtype: SynapseDtype | np.dtype = dtype
        self.grad_fn: SynapseBackwardPropNode | None = grad_fn
        self.grad = np.zeros_like(
            data,
            (
                _SYNAPSE_DTYPE_TO_NP_DTYPE_MAP[dtype]
                if not isinstance(dtype, np.dtype)
                else dtype
            ),
        )
        self._prev = [] # Maintain parents for toposort

    def numpy(self):
        return self.data

    def __repr__(self) -> str:
        return (
            f"Tensor [ Data: {self.data}"
            + f" | Dtype: {self.dtype}"
            + f" | Shape: {list(self.data.shape)} ]"
        )

    @staticmethod
    def _apply(op_cls, *input_tensors):
        ctx = SynapseComputeContext()
        tensor_nps = [input_tensor.numpy() for input_tensor in input_tensors]
        
        # Forward pass
        out_result_data = op_cls.forward(ctx, *tensor_nps)
        any_requires_grad = any(
            input_tensor.requires_grad for input_tensor in input_tensors
        )
        out_tensor = SynapseTensor(
            out_result_data,
            shape=out_result_data.shape,
            dtype=out_result_data.dtype,
            requires_grad=any_requires_grad,
            grad_fn=None,
        )
        
        if any_requires_grad:
            out_tensor.grad_fn = SynapseBackwardPropNode(
                op_cls,
                ctx,
                input_tensors,
            )
            out_tensor._prev = [input_tensor for input_tensor in input_tensors if input_tensor.requires_grad]
            
        return out_tensor
    


    def __add__(self, other):
        return SynapseTensor._apply(Add, self, other)


_SYNAPSE_DTYPE_TO_NP_DTYPE_MAP = {
    SynapseDtype.float32: np.float32,
    SynapseDtype.float64: np.float64,
    SynapseDtype.int32: np.int32,
    SynapseDtype.int64: np.int64,
}
