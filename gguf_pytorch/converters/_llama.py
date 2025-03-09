import torch
from torch import Tensor
from transformers import LlamaConfig, LlamaForCausalLM

from .utils import load_state_dict, merge_weights


def covert_llama(
    metadata: dict[str, int | float | str],
    gguf_state_dict: dict[str, Tensor],
    format: str,
) -> tuple[LlamaConfig, dict[str, Tensor]]:
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

    # need to manually add this. HF is too dumb to include this automatically when serialize to JSON.
    config.architectures = [LlamaForCausalLM.__name__]

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

    dtype = gguf_state_dict["token_embd.weight"].dtype

    pt_state_dict = dict()
    for k, v in gguf_state_dict.items():
        # llama.cpp uses some strange layout
        if k.endswith(".attn_q.weight"):
            v = v.view(config.num_attention_heads, -1, 2, config.hidden_size).transpose(1, 2).reshape(v.shape)
        elif k.endswith(".attn_k.weight"):
            v = v.view(config.num_key_value_heads, -1, 2, config.hidden_size).transpose(1, 2).reshape(v.shape)

        # llama.cpp uses FP32 for norm weight+bias and linear bias
        elif k.endswith("norm.weight"):
            v = v.to(dtype)

        pt_state_dict[map_key(k)] = v

    pt_state_dict.pop("rope_freqs.weight")  # TODO: check this

    if format == "vllm":
        merge_weights(pt_state_dict, ["q_proj", "k_proj", "v_proj"], "qkv_proj")
        merge_weights(pt_state_dict, ["gate_proj", "up_proj"], "gate_up_proj")
    else:
        assert format == "hf"

    return config, pt_state_dict


def load_llama(config: LlamaConfig, state_dict: dict[str, Tensor]):
    with torch.device("meta"):
        model = LlamaForCausalLM(config)

    load_state_dict(model, state_dict, assign=True)
    return model
