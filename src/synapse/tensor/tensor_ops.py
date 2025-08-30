from synapse.autodiff.context import SynapseComputeContext

class SynapseTensorOp:
    @staticmethod
    def forward(ctx, *inputs):
        raise NotImplementedError()

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplementedError()

class Add(SynapseTensorOp):
    @staticmethod
    def forward(ctx: SynapseComputeContext, a, b):
        ctx.cache_for_backward(a.shape, b.shape)
        return a + b

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, grad_out
