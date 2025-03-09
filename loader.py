# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import math
import struct
import typing
from enum import Enum, auto

import numpy as np
import torch

from converters import CONVERTER_LOOKUP, LOADER_LOOKUP


# https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
class GGML_TYPE(Enum):
    F32 = 0  # auto() starts with 1
    F16 = auto()
    Q4_0 = auto()
    Q4_1 = auto()
    Q4_2 = auto()  # (removed)
    Q4_3 = auto()  # (removed)
    Q5_0 = auto()
    Q5_1 = auto()
    Q8_0 = auto()
    Q8_1 = auto()
    Q2_K = auto()
    Q3_K = auto()
    Q4_K = auto()
    Q5_K = auto()
    Q6_K = auto()
    Q8_K = auto()
    IQ2_XXS = auto()
    IQ2_XS = auto()
    IQ3_XXS = auto()
    IQ1_S = auto()
    IQ4_NL = auto()
    IQ3_S = auto()
    IQ2_S = auto()
    IQ4_XS = auto()
    I8 = auto()
    I16 = auto()
    I32 = auto()
    I64 = auto()
    F64 = auto()
    IQ1_M = auto()
    BF16 = auto()
    Q4_0_4_4 = auto()  # (removed)
    Q4_0_4_8 = auto()  # (removed)
    Q4_0_8_8 = auto()  # (removed)
    TQ1_0 = auto()
    TQ2_0 = auto()
    IQ4_NL_4_4 = auto()  # (removed)
    IQ4_NL_4_8 = auto()  # (removed)
    IQ4_NL_8_8 = auto()  # (removed)


def _decode_number(f: typing.BinaryIO, dtype: str):
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


def _decode_str(f: typing.BinaryIO):
    return f.read(_decode_number(f, "u64")).decode()


def _decode_metadata_value(f: typing.BinaryIO, value_type: int | None = None):
    if value_type is None:
        value_type = _decode_number(f, "u32")

    lookup = [
        "u8",
        "i8",
        "u16",
        "i16",
        "u32",
        "i32",
        "f32",
        "u8",  # bool
        "str",
        "array",
        "u64",
        "i64",
        "f64",
    ]
    assert value_type < len(lookup), value_type
    value_type = lookup[value_type]

    if value_type == "str":
        value = _decode_str(f)
    elif value_type == "array":
        elem_type = _decode_number(f, "u32")
        count = _decode_number(f, "u64")
        value = [_decode_metadata_value(f, elem_type) for _ in range(count)]
    else:
        value = _decode_number(f, value_type)

    return value


def load_gguf(filename: str, format: str = "gguf"):
    f = open(filename, "rb")

    assert (magic_number := f.read(4)) == b"GGUF", magic_number
    assert (version := _decode_number(f, "u32")) == 3, version
    num_tensors = _decode_number(f, "u64")
    num_metadata = _decode_number(f, "u64")

    metadata = dict()
    for _ in range(num_metadata):
        key = _decode_str(f)
        value = _decode_metadata_value(f)
        metadata[key] = value

    state_dict_meta = dict()
    for _ in range(num_tensors):
        name = _decode_str(f)
        ndim = _decode_number(f, "u32")
        shape = [_decode_number(f, "u64") for _ in range(ndim)][::-1]  # shape order is reversed in GGML
        ggml_type = GGML_TYPE(_decode_number(f, "u32"))
        offset = _decode_number(f, "u64")

        state_dict_meta[name] = (shape, ggml_type, offset)

    alignment = metadata.get("general.alignment", 32)
    base_offset = (f.tell() + alignment - 1) // alignment * alignment
    f.close()

    state_dict = dict()
    tensor_data = torch.from_numpy(np.memmap(filename, mode="r", offset=base_offset))

    for name, (shape, ggml_type, offset) in state_dict_meta.items():
        numel = math.prod(shape)

        basic_dtype_lookup = {
            GGML_TYPE.F64: torch.float64,
            GGML_TYPE.F32: torch.float32,
            GGML_TYPE.F16: torch.float16,
            GGML_TYPE.BF16: torch.bfloat16,
            GGML_TYPE.I8: torch.int8,
            GGML_TYPE.I16: torch.int16,
            GGML_TYPE.I32: torch.int32,
            GGML_TYPE.I64: torch.int64,
        }

        if ggml_type not in basic_dtype_lookup:
            raise ValueError(f"Unsupported {ggml_type=}")

        dtype = basic_dtype_lookup[ggml_type]
        tensor = tensor_data[offset : offset + numel * dtype.itemsize].view(dtype).view(shape)
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
