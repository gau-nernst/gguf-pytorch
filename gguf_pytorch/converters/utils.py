import torch
from torch import Tensor, nn


def merge_weights(state_dict: dict[str, Tensor], old_labels: list[str], new_label: str):
    # NOTE: this is done in-place
    new_names = []

    for name in list(state_dict.keys()):
        for i, old_label in enumerate(old_labels):
            if old_label in name.split("."):
                prefix, suffix = name.split(old_label)  # suffix is either weight or bias
                new_name = f"{prefix}{new_label}{suffix}"

                if new_name not in state_dict:
                    state_dict[new_name] = [None] * len(old_labels)
                    new_names.append(new_name)

                state_dict[new_name][i] = state_dict.pop(name)
                break

    for name in new_names:
        state_dict[name] = torch.cat(state_dict[name], dim=0)

    return state_dict


def load_state_dict(model: nn.Module, state_dict: dict[str, Tensor], assign: bool = False):
    """Load state dict with tied weights handling"""
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, assign=assign, strict=False)

    if model.config.tie_word_embeddings:
        if hasattr(model, "tie_weights"):  # HF
            model.tie_weights()
        elif hasattr(model.lm_head, "tie_weights"):  # vLLM
            model.lm_head.tie_weights(model.embed_tokens)
        else:
            raise RuntimeError(f"Unsupported {model.__class__=}")
        missing_keys.remove("lm_head.weight")

    assert len(missing_keys) == 0 and len(unexpected_keys) == 0
