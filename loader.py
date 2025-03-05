# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import math
import struct
import typing

import numpy as np
import torch

# https://github.com/ggml-org/llama.cpp/blob/master/ggml/include/ggml.h
GGML_TYPE_LOOKUP = [
    "GGML_TYPE_F32",
    "GGML_TYPE_F16",
    "GGML_TYPE_Q4_0",
    "GGML_TYPE_Q4_1",
    "GGML_TYPE_Q4_2",  # (removed)
    "GGML_TYPE_Q4_3",  # (removed)
    "GGML_TYPE_Q5_0",
    "GGML_TYPE_Q5_1",
    "GGML_TYPE_Q8_0",
    "GGML_TYPE_Q8_1",
    "GGML_TYPE_Q2_K",
    "GGML_TYPE_Q3_K",
    "GGML_TYPE_Q4_K",
    "GGML_TYPE_Q5_K",
    "GGML_TYPE_Q6_K",
    "GGML_TYPE_Q8_K",
    "GGML_TYPE_IQ2_XXS",
    "GGML_TYPE_IQ2_XS",
    "GGML_TYPE_IQ3_XXS",
    "GGML_TYPE_IQ1_S",
    "GGML_TYPE_IQ4_NL",
    "GGML_TYPE_IQ3_S",
    "GGML_TYPE_IQ2_S",
    "GGML_TYPE_IQ4_XS",
    "GGML_TYPE_I8",
    "GGML_TYPE_I16",
    "GGML_TYPE_I32",
    "GGML_TYPE_I64",
    "GGML_TYPE_F64",
    "GGML_TYPE_IQ1_M",
    "GGML_TYPE_BF16",
    "GGML_TYPE_Q4_0_4_4",  # (removed)
    "GGML_TYPE_Q4_0_4_8",  # (removed)
    "GGML_TYPE_Q4_0_8_8",  # (removed)
    "GGML_TYPE_TQ1_0",
    "GGML_TYPE_TQ2_0",
    "GGML_TYPE_IQ4_NL_4_4",  # (removed)
    "GGML_TYPE_IQ4_NL_4_8",  # (removed)
    "GGML_TYPE_IQ4_NL_8_8",  # (removed)
]


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


def load_gguf(filename: str):
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
        ggml_type = GGML_TYPE_LOOKUP[_decode_number(f, "u32")]
        offset = _decode_number(f, "u64")

        state_dict_meta[name] = (shape, ggml_type, offset)

    alignment = metadata.get("general.alignment", 32)
    base_offset = (f.tell() + alignment - 1) // alignment * alignment
    f.close()

    state_dict = dict()
    tensor_data = torch.from_numpy(np.memmap(filename, mode="r", offset=base_offset))

    for name, (shape, ggml_type, offset) in state_dict_meta.items():
        numel = math.prod(shape)

        basic_dtype_lookup = dict(
            GGML_TYPE_F64=torch.float64,
            GGML_TYPE_F32=torch.float32,
            GGML_TYPE_F16=torch.float16,
            GGML_TYPE_BF16=torch.bfloat16,
            GGML_TYPE_I8=torch.int8,
            GGML_TYPE_I16=torch.int16,
            GGML_TYPE_I32=torch.int32,
            GGML_TYPE_I64=torch.int64,
        )

        if ggml_type not in basic_dtype_lookup:
            raise ValueError(f"Unsupported {ggml_type=}")

        dtype = basic_dtype_lookup[ggml_type]
        tensor = tensor_data[offset : offset + numel * dtype.itemsize].view(dtype).view(shape)
        state_dict[name] = tensor

    return metadata, state_dict
