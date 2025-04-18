# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import logging
import math
import struct
import typing

import numpy as np
import torch

from .constants import GGML_TYPE
from .converters import CONVERTER_LOOKUP, LOADER_LOOKUP
from .ggml_tensor import GGMLTensor

logger = logging.getLogger(__name__)


def _read_number(f: typing.BinaryIO, dtype: str):
    format, nbyte = dict(
        i8=("<b", 1),
        u8=("<B", 1),
        i16=("<h", 2),
        u16=("<H", 2),
        i32=("<l", 4),
        u32=("<L", 4),
        i64=("<q", 8),
        u64=("<Q", 8),
        f32=("<f", 4),
        f64=("<d", 8),
    )[dtype]
    return struct.unpack_from(format, f.read(nbyte))[0]


def _read_str(f: typing.BinaryIO):
    return f.read(_read_number(f, "u64")).decode()


def _read_metadata_value(f: typing.BinaryIO, value_type: int | None = None):
    if value_type is None:
        value_type = _read_number(f, "u32")

    lookup = [
        "u8",
        "i8",
        "u16",
        "i16",
        "u32",
        "i32",
        "f32",
        "bool",
        "str",
        "array",
        "u64",
        "i64",
        "f64",
    ]
    assert value_type < len(lookup), value_type
    value_type = lookup[value_type]

    if value_type == "str":
        value = _read_str(f)
    elif value_type == "array":
        elem_type = _read_number(f, "u32")
        count = _read_number(f, "u64")
        value = [_read_metadata_value(f, elem_type) for _ in range(count)]
    elif value_type == "bool":
        value = bool(_read_number(f, "u8"))
    else:
        value = _read_number(f, value_type)

    return value


def load_gguf(filename: str, format: str = "gguf", skip_unsupported: bool = False):
    f = open(filename, "rb")

    assert (magic_number := f.read(4)) == b"GGUF", magic_number
    assert (version := _read_number(f, "u32")) == 3, version
    num_tensors = _read_number(f, "u64")
    num_metadata = _read_number(f, "u64")

    metadata = dict()
    for _ in range(num_metadata):
        key = _read_str(f)
        value = _read_metadata_value(f)
        metadata[key] = value

    state_dict_meta = dict()
    for _ in range(num_tensors):
        name = _read_str(f)
        ndim = _read_number(f, "u32")
        shape = [_read_number(f, "u64") for _ in range(ndim)][::-1]  # shape order is reversed in GGML
        ggml_type = GGML_TYPE(_read_number(f, "u32"))
        offset = _read_number(f, "u64")

        state_dict_meta[name] = (shape, ggml_type, offset)

    alignment = metadata.get("general.alignment", 32)
    base_offset = (f.tell() + alignment - 1) // alignment * alignment
    f.close()

    if num_tensors == 0:
        return metadata, dict()

    state_dict = dict()
    tensor_data = torch.from_numpy(np.memmap(filename, mode="r", offset=base_offset))

    for name, (shape, ggml_type, offset) in state_dict_meta.items():
        numel = math.prod(shape)

        BASIC_TYPE_LOOKUP = {
            GGML_TYPE.F64: torch.float64,
            GGML_TYPE.F32: torch.float32,
            GGML_TYPE.F16: torch.float16,
            GGML_TYPE.BF16: torch.bfloat16,
            GGML_TYPE.I8: torch.int8,
            GGML_TYPE.I16: torch.int16,
            GGML_TYPE.I32: torch.int32,
            GGML_TYPE.I64: torch.int64,
        }

        if ggml_type in BASIC_TYPE_LOOKUP:
            dtype = BASIC_TYPE_LOOKUP[ggml_type]
            tensor = tensor_data[offset : offset + numel * dtype.itemsize].view(dtype).view(shape)

        else:
            try:
                tensor = GGMLTensor.from_buffer(tensor_data[offset:], ggml_type, shape)
            except Exception as e:
                msg = f"Fail to convert {name} with {ggml_type}"
                if skip_unsupported:
                    logger.warning(msg)
                else:
                    raise RuntimeError(msg) from e

        state_dict[name] = tensor

    if format != "gguf":
        converter = CONVERTER_LOOKUP[metadata["general.architecture"]]
        metadata, state_dict = converter(metadata, state_dict, format)

    return metadata, state_dict


def load_gguf_model(filename: str):
    metadata, state_dict = load_gguf(filename, format="hf")

    normal_dtype = torch.bfloat16 if any(v.dtype == torch.bfloat16 for v in state_dict.values()) else torch.float16

    for k, v in state_dict.items():
        if k.endswith("norm.weight"):
            state_dict[k] = v.to(normal_dtype)

    loader = LOADER_LOOKUP[metadata["general.architecture"]]
    model = loader(metadata, state_dict)

    return model
