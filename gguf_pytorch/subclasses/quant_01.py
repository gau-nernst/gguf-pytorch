import math

import torch
import torch.nn.functional as F
from torch import Tensor

from ..constants import GGML_TYPE

# https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-common.h


def _dequantize(int_data: Tensor, scales: Tensor, offsets: Tensor | None, ggml_type: GGML_TYPE):
    if ggml_type == GGML_TYPE.Q8_0:
        out = int_data.view(torch.int8).float().reshape(-1, Quant01.BLOCK_SIZE) * scales.reshape(-1, 1)

    elif ggml_type == GGML_TYPE.Q4_0:
        data = torch.stack([int_data & 0xF, int_data >> 4], dim=-2).float()
        out = (data - 8).reshape(-1, Quant01.BLOCK_SIZE) * scales.reshape(-1, 1)

    elif ggml_type == GGML_TYPE.Q4_1:
        data = torch.stack([int_data & 0xF, int_data >> 4], dim=-2).float()
        out = data.reshape(-1, Quant01.BLOCK_SIZE) * scales.reshape(-1, 1) + offsets.reshape(-1, 1)

    return out.to(scales.dtype)


def _reshape(x: "Quant01"):
    shape = list(x.shape)
    if x.ggml_type in (GGML_TYPE.Q4_0, GGML_TYPE.Q4_1):
        shape[-1] //= 2
    int_data = x.int_data.flatten().view(shape)

    shape = x.shape[:-1] + (shape[-1] // x.BLOCK_SIZE,)
    scales = x.scales.flatten().view(shape)
    offsets = x.offsets.flatten().view(shape) if x.offsets is not None else None

    return int_data, scales, offsets


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

    def __tensor_flatten__(self):
        if self.offsets is not None:
            return ["int_data", "scales", "offsets"], [self.ggml_type, self.shape]
        else:
            return ["int_data", "scales"], [self.offsets, self.ggml_type, self.shape]

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size=None, outer_stride=None):
        return cls(*tensor_data_dict.values(), *tensor_attributes)

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
        return _dequantize(self.int_data, self.scales, self.offsets, self.ggml_type).view(self.shape)

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
            int_data, scales, offsets = _reshape(weight)
            out = _dequantize(
                F.embedding(input, int_data),
                F.embedding(input, scales),
                F.embedding(input, offsets) if offsets is not None else None,
                weight.ggml_type,
            )
            return out.view(input.shape + weight.shape[1:])

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

            assert shape[-1] % Quant01.BLOCK_SIZE == 0
            return Quant01(x.int_data, x.scales, x.offsets, x.ggml_type, tuple(shape))

        elif func is aten.transpose.int:
            x: Quant01 = args[0]
            dim0 = args[1]
            dim1 = args[2]

            int_data, scales, offsets = _reshape(x)
            int_data = int_data.transpose(dim0, dim1)
            scales = scales.transpose(dim0, dim1)
            offsets = offsets.transpose(dim0, dim1) if offsets is not None else None

            shape = list(x.shape)
            shape[dim0], shape[dim1] = shape[dim1], shape[dim0]
            return Quant01(int_data, scales, offsets, x.ggml_type, shape)

        elif func is aten.cat.default:
            tensors = args[0]

            assert all(isinstance(x, Quant01) for x in tensors)
            assert all(x.ggml_type == tensors[0].ggml_type for x in tensors)

            int_data_list, scales_list, offsets_list = zip(*[_reshape(x) for x in tensors])
            int_data = func(int_data_list, *args[1:])
            scales = func(scales_list, *args[1:])
            offsets = func(offsets_list, *args[1:]) if offsets_list[0] is not None else None

            shape = list(scales.shape)
            shape[-1] *= Quant01.BLOCK_SIZE
            return Quant01(int_data, scales, offsets, tensors[0].ggml_type, shape)

        elif func is aten.detach.default:
            x: Quant01 = args[0]
            return Quant01(x.int_data, x.scales, x.offsets, x.ggml_type, x.shape)

        elif func is aten._to_copy.default:
            x: Quant01 = args[0]
            dtype = kwargs.get("dtype", None)
            device = kwargs.get("device", None)
            return Quant01(
                x.int_data.to(device=device),
                x.scales.to(device=device, dtype=dtype),
                x.offsets.to(device=device, dtype=dtype) if x.offsets is not None else None,
                x.ggml_type,
                x.shape,
            )

        msg = f"{cls.__name__} dispatch: {func} is not implemented"
        for i, arg in enumerate(args):
            msg += f"\n- args[{i}]={arg}"
        for k, v in kwargs.items():
            msg += f"\n- {k}={v}"
        raise NotImplementedError(msg)
