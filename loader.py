# https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

import math
import struct
import typing
from enum import Enum, auto

import numpy as np
import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM


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

    return metadata, state_dict


def _load_llama(metadata: dict[str, int | float | str], gguf_state_dict: dict[str, Tensor]):
    config = LlamaConfig(
        max_position_embeddings=metadata["llama.context_length"],
        hidden_size=metadata["llama.embedding_length"],
        num_hidden_layers=metadata["llama.block_count"],
        intermediate_size=metadata["llama.feed_forward_length"],
        num_attention_heads=metadata["llama.attention.head_count"],
        num_key_value_heads=metadata["llama.attention.head_count_kv"],
        rms_norm_eps=metadata["llama.attention.layer_norm_rms_epsilon"],
        rope_theta=metadata["llama.rope.freq_base"],
        head_dim=metadata["llama.attention.key_length"],
        vocab_size=metadata["llama.vocab_size"],
        tie_word_embeddings="output.weight" not in gguf_state_dict,
    )

    def map_key(k: str):
        k = k.replace("token_embd.", "model.embed_tokens.")
        k = k.replace("blk.", "model.layers.")
        k = k.replace(".attn_norm.", ".input_layernorm.")
        k = k.replace(".attn_q.", ".self_attn.q_proj.")
        k = k.replace(".attn_k.", ".self_attn.k_proj.")
        k = k.replace(".attn_v.", ".self_attn.v_proj.")
        k = k.replace(".attn_output.", ".self_attn.o_proj.")
        k = k.replace(".ffn_norm.", ".post_attention_layernorm.")
        k = k.replace(".ffn_up.", ".mlp.up_proj.")
        k = k.replace(".ffn_gate.", ".mlp.gate_proj.")
        k = k.replace(".ffn_down.", ".mlp.down_proj.")
        k = k.replace("output_norm.", "model.norm.")
        return k

    with torch.device("meta"):
        model = LlamaForCausalLM(config)

    pt_state_dict = dict()
    for k, v in gguf_state_dict.items():
        if k.endswith(".attn_q.weight"):
            v = v.view(config.num_attention_heads, -1, 2, config.hidden_size).transpose(1, 2).reshape(v.shape)
        elif k.endswith((".attn_k.weight", ".attn_v.weight")):
            v = v.view(config.num_key_value_heads, -1, 2, config.hidden_size).transpose(1, 2).reshape(v.shape)
        pt_state_dict[map_key(k)] = v

    missing_keys, unexpected_keys = model.load_state_dict(pt_state_dict, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.tie_weights()
        missing_keys.remove("lm_head.weight")

    unexpected_keys.remove("rope_freqs.weight")  # TODO: check this
    assert len(missing_keys) == 0 and len(unexpected_keys) == 0

    return model


def load_gguf_model(filename: str):
    metadata, state_dict = load_gguf(filename)
    arch = metadata["general.architecture"]

    normal_dtype = torch.bfloat16 if any(v.dtype == torch.bfloat16 for v in state_dict.values()) else torch.float16

    for k, v in state_dict.items():
        if k.endswith("_norm.weight"):
            state_dict[k] = v.to(normal_dtype)

    if arch == "llama":
        model = _load_llama(metadata, state_dict)

    else:
        raise RuntimeError(f"Unsupported {arch=}")

    return model
