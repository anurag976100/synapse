# See https://en.wikipedia.org/wiki/Finite_difference_method

from contextlib import contextmanager

import numpy as np


class SynapseBackwardPropNode:
    def __init__(
        self, op_cls, ctx, inputs
    ):
        self.op_cls = op_cls
        self.ctx = ctx
        self.inputs = inputs