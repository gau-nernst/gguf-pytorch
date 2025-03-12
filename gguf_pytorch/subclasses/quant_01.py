import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..constants import GGML_TYPE

# https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h
BLOCK_SIZE = 32
LAYOUT_MAP = {
    GGML_TYPE.Q8_0: [2, BLOCK_SIZE],
    GGML_TYPE.Q4_0: [2, BLOCK_SIZE // 2],
    GGML_TYPE.Q4_1: [2, 2, BLOCK_SIZE // 2],
}


def _dequantize(buffer: Tensor, ggml_type: GGML_TYPE) -> Tensor:
    layout = LAYOUT_MAP[ggml_type]
    shape = list(buffer.shape)
    shape[-1] = shape[-1] // sum(layout) * BLOCK_SIZE

    buffer = buffer.reshape(-1, sum(layout))

    if ggml_type == GGML_TYPE.Q8_0:
        scales, int_data = buffer.split(layout, dim=1)
        out = int_data.view(torch.int8).float() * scales.view(torch.float16)

    elif ggml_type == GGML_TYPE.Q4_0:
        scales, int_data = buffer.split(layout, dim=1)
        data = torch.stack([int_data & 0xF, int_data >> 4], dim=-2).float()
        out = (data - 8).reshape(-1, BLOCK_SIZE) * scales.view(torch.float16)

    elif ggml_type == GGML_TYPE.Q4_1:
        scales, offsets, int_data = buffer.split(layout, dim=1)
        data = torch.stack([int_data & 0xF, int_data >> 4], dim=-2).float()
        out = data.reshape(-1, BLOCK_SIZE) * scales.view(torch.float16) + offsets.view(torch.float16)

    return out.to(torch.float16).view(shape)


def _reshape(buffer: Tensor, ggml_type: GGML_TYPE, shape: tuple[int, ...]) -> Tensor:
    assert shape[-1] % BLOCK_SIZE == 0
    nbytes_per_block = sum(LAYOUT_MAP[ggml_type])
    buffer_shape = list(shape[:-1]) + [shape[-1] // BLOCK_SIZE * nbytes_per_block]
    return buffer.reshape(buffer_shape)


class Quant01(Tensor):
    @staticmethod
    def __new__(cls, buffer: Tensor, ggml_type: GGML_TYPE):
        shape = list(buffer.shape)
        nbytes_per_block = sum(LAYOUT_MAP[ggml_type])
        shape[-1] = shape[-1] // nbytes_per_block * BLOCK_SIZE
        return Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=torch.float16,
            device=buffer.device,
        )

    def __init__(self, buffer: Tensor, ggml_type: GGML_TYPE) -> None:
        assert buffer.dtype is torch.uint8
        self.buffer = buffer
        self.ggml_type = ggml_type

        # TODO: add kernel backend e.g. tinygemm, gemlite. then, preprocess the weights to the one required by that backend.

    def __tensor_flatten__(self):
        return ["buffer"], [self.ggml_type]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*tensor_data_dict.values(), *tensor_attributes)

    @staticmethod
    def from_buffer(buffer: Tensor, ggml_type: GGML_TYPE, shape: tuple[int, ...]):
        assert buffer.dtype == torch.uint8

        nbytes_per_block = sum(LAYOUT_MAP[ggml_type])
        nbytes = math.prod(shape) // BLOCK_SIZE * nbytes_per_block
        buffer = _reshape(buffer[:nbytes], ggml_type, shape)
        return Quant01(buffer, ggml_type)

    def dequantize(self):
        return _dequantize(self.buffer, self.ggml_type)

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

        elif func is F.embedding:
            input: Tensor = args[0]
            weight: Quant01 = args[1]
            return _dequantize(F.embedding(input, weight.buffer), weight.ggml_type)

        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        aten = torch.ops.aten

        if func is aten.view.default:
            x: Quant01 = args[0]
            shape = list(args[1])

            if -1 in shape:
                idx = shape.index(-1)
                shape[idx] = 1
                assert x.numel() % math.prod(shape) == 0
                shape[idx] = x.numel() // math.prod(shape)
            else:
                assert math.prod(shape) == x.numel()

            return Quant01(_reshape(x.buffer, x.ggml_type, shape), x.ggml_type)

        elif func is aten.transpose.int:
            x: Quant01 = args[0]
            dim0 = args[1]
            dim1 = args[2]

            # if transpose involves the last dimension, we have to requantize
            assert dim0 not in (-1, x.ndim - 1)
            assert dim1 not in (-1, x.ndim - 1)
            return Quant01(x.buffer.transpose(dim0, dim1), x.ggml_type)

        elif func is aten.cat.default:
            tensors: list[Quant01] = args[0]

            assert all(isinstance(x, Quant01) for x in tensors)
            assert all(x.ggml_type == tensors[0].ggml_type for x in tensors)
            ggml_type = tensors[0].ggml_type

            buffer_list = [_reshape(x.buffer, ggml_type, x.shape) for x in tensors]
            buffer = func(buffer_list, *args[1:])
            return Quant01(buffer, ggml_type)

        elif func is aten.detach.default:
            x: Quant01 = args[0]
            return Quant01(x.buffer, x.ggml_type)

        elif func is aten._to_copy.default:
            x: Quant01 = args[0]
            # NOTE: we don't support changing dtype
            device = kwargs.get("device", None)
            return Quant01(x.buffer.to(device=device), x.ggml_type)

        msg = f"{cls.__name__} dispatch: {func} is not implemented"
        for i, arg in enumerate(args):
            msg += f"\n- args[{i}]={arg}"
        for k, v in kwargs.items():
            msg += f"\n- {k}={v}"
        raise NotImplementedError(msg)
