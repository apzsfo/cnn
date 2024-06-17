from .layers import *

def affine_relu_forward(x, w, b):
    """
    Performs an affine transform followed by a ReLU activation.
    
    Inputs:
    - x: Input data to the affine layer (N, D) where N is the number of samples and D is the number of features.
    - w: Weights for the affine layer (D, M) where M is the number of output units.
    - b: Biases for the affine layer (M,)
    
    Returns:
    - out: Output from the ReLU activation (N, M)
    - cache: Tuple containing intermediate values needed for the backward pass
    """
    # Affine forward pass
    a, fc_cache = affine_forward(x, w, b)
    
    # ReLU forward pass
    out, relu_cache = relu_forward(a)
    
    # Cache both affine and ReLU caches for backward pass
    cache = (fc_cache, relu_cache)
    
    return out, cache

def affine_relu_backward(dout, cache):
    """
    Performs the backward pass for the affine-relu convenience layer.
    
    Inputs:
    - dout: Upstream derivatives (N, M)
    - cache: Tuple containing intermediate values from the forward pass
    
    Returns:
    - dx: Gradient with respect to input x (N, D)
    - dw: Gradient with respect to weights w (D, M)
    - db: Gradient with respect to biases b (M,)
    """
    # Unpack cache
    fc_cache, relu_cache = cache
    
    # ReLU backward pass
    da = relu_backward(dout, relu_cache)
    
    # Affine backward pass
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db
