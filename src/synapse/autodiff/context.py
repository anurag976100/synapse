class SynapseComputeContext:
    def __init__(self):
        self.saved_tensor_shapes = ()
        
    def cache_for_backward(self, *tensor_shapes):
        self.saved_tensors = tensor_shapes