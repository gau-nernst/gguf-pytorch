import torch
from torch import Tensor


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
