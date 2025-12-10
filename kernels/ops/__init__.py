from .base import BaseOp, OpConfig, OP_REGISTRY, register_op


# import all Op here to trigger register_op
from .vec_add import VecAddOp
from .softmax import SoftmaxOp