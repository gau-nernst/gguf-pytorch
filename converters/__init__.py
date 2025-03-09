from ._llama import covert_llama, load_llama

CONVERTER_LOOKUP = dict(
    llama=covert_llama,
)
LOADER_LOOKUP = dict(
    llama=load_llama,
)
