import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM

from .utils import merge_weights


def convert_llama_state_dict(
    metadata: dict[str, int | float | str],
    gguf_state_dict: dict[str, Tensor],
    format: str,
) -> dict[str, Tensor]:
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
        k = k.replace("output.weight", "lm_head.weight")
        return k

    hidden_size = metadata["llama.embedding_length"]
    num_q_heads = metadata["llama.attention.head_count"]
    num_kv_heads = metadata["llama.attention.head_count_kv"]

    pt_state_dict = dict()
    for k, v in gguf_state_dict.items():
        # llama.cpp uses some strange layout
        if k.endswith(".attn_q.weight"):
            v = v.view(num_q_heads, -1, 2, hidden_size).transpose(1, 2).reshape(v.shape)
        elif k.endswith(".attn_k.weight"):
            v = v.view(num_kv_heads, -1, 2, hidden_size).transpose(1, 2).reshape(v.shape)
        pt_state_dict[map_key(k)] = v

    pt_state_dict.pop("rope_freqs.weight")  # TODO: check this

    if format == "vllm":
        merge_weights(pt_state_dict, ["q_proj", "k_proj", "v_proj"], "qkv_proj")
        merge_weights(pt_state_dict, ["gate_proj", "up_proj"], "gate_up_proj")
    else:
        assert format == "hf"

    return pt_state_dict


def load_llama(metadata: dict[str, int | float | str], state_dict: dict[str, Tensor]):
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
        tie_word_embeddings="lm_head.weight" not in state_dict,
    )

    with torch.device("meta"):
        model = LlamaForCausalLM(config)

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.tie_weights()
        missing_keys.remove("lm_head.weight")

    assert len(missing_keys) == 0 and len(unexpected_keys) == 0

    return model
