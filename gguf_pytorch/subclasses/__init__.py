from ..constants import GGML_TYPE
from .quant_01 import Quant01

SUBCLASS_TYPE_LOOKUP = {
    GGML_TYPE.Q8_0: Quant01,
    GGML_TYPE.Q4_0: Quant01,
    GGML_TYPE.Q4_1: Quant01,
}
