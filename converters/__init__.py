from ._llama import convert_llama_state_dict, load_llama

CONVERTER_LOOKUP = dict(
    llama=convert_llama_state_dict,
)
LOADER_LOOKUP = dict(
    llama=load_llama,
)
