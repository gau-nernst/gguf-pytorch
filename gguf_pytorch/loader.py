# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import logging
import math
import struct
import typing

import numpy as np
import torch

from .constants import GGML_TYPE
from .converters import CONVERTER_LOOKUP, LOADER_LOOKUP
from .subclasses import SUBCLASS_TYPE_LOOKUP

logger = logging.getLogger(__name__)


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


def load_gguf(filename: str, format: str = "gguf", skip_unsupported: bool = False):
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

        elif ggml_type in SUBCLASS_TYPE_LOOKUP:
            subclass = SUBCLASS_TYPE_LOOKUP[ggml_type]
            tensor = subclass.from_buffer(tensor_data[offset:], ggml_type, shape)

        else:
            msg = f"Param {name} uses unsupported {ggml_type}"
            if skip_unsupported:
                logger.warning(msg)
                continue
            else:
                raise ValueError()

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
