import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..constants import GGML_TYPE

# https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h


class Quant01(Tensor):
    BLOCK_SIZE = 32

    @staticmethod
    # @torch._dynamo.disable
    def __new__(
        cls,
        int_data: Tensor,
        scales: Tensor,
        offsets: Tensor | None,
        ggml_type: GGML_TYPE,
        shape: tuple[int, ...],
    ):
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=scales.dtype,
            device=int_data.device,
        )

    # @torch._dynamo.disable
    def __init__(
        self,
        int_data: Tensor,
        scales: Tensor,
        offsets: Tensor | None,
        ggml_type: GGML_TYPE,
        shape: tuple[int, ...],
    ):
        assert int_data.dtype is torch.uint8
        self.int_data = int_data
        self.scales = scales
        self.offsets = offsets
        self.ggml_type = ggml_type

        # TODO: add kernel backend e.g. tinygemm, gemlite. then, preprocess the weights to the one required by that backend.

    @staticmethod
    def from_buffer(buffer: Tensor, ggml_type: GGML_TYPE, shape: tuple[int, ...]):
        assert buffer.dtype == torch.uint8

        layout = {
            GGML_TYPE.Q8_0: [2, Quant01.BLOCK_SIZE],
            GGML_TYPE.Q4_0: [2, Quant01.BLOCK_SIZE // 2],
            GGML_TYPE.Q4_1: [2, 2, Quant01.BLOCK_SIZE // 2],
        }[ggml_type]

        nbytes = math.prod(shape) // Quant01.BLOCK_SIZE * sum(layout)
        buffer = buffer[:nbytes].view(-1, sum(layout))

        if len(layout) == 2:
            scales, int_data = buffer.split(layout, dim=1)
            offsets = None
        else:
            scales, offsets, int_data = buffer.split(layout, dim=1)
            offsets = offsets.view(torch.float16)

        scales = scales.view(torch.float16)
        return Quant01(int_data, scales, offsets, ggml_type, shape)

    def dequantize(self):
        if self.ggml_type == GGML_TYPE.Q8_0:
            out = self.int_data.view(torch.int8).float().reshape(-1, Quant01.BLOCK_SIZE) * self.scales.reshape(-1, 1)

        elif self.ggml_type == GGML_TYPE.Q4_0:
            data = self.int_data
            data = torch.stack([data & 0xF, data >> 4], dim=-2).float()
            out = (data - 8).reshape(-1, Quant01.BLOCK_SIZE) * self.scales.reshape(-1, 1)

        elif self.ggml_type == GGML_TYPE.Q4_1:
            data = self.int_data
            data = torch.stack([data & 0xF, data >> 4], dim=-2).float()
            out = data.reshape(-1, Quant01.BLOCK_SIZE) * self.scales.reshape(-1, 1) + self.offsets.reshape(-1, 1)

        return out.reshape(self.shape)

    def __repr__(self):
        fields = dict(
            ggml_type=self.ggml_type,
            shape=tuple(self.shape),
            device=self.device,
        )
        fields_str = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"{self.__class__.__name__}({fields_str})"

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or dict()

        if func is F.linear:
            x: Tensor = args[0]
            w: Quant01 = args[1]
            b: Tensor | None = args[2] if len(args) > 2 else None
            return F.linear(x, w.dequantize(), b)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        raise NotImplementedError(f"{cls.__name__} dispatch: {func} is not implemented")
