import numpy as np

def affine_transform_forward(input_data, weight_matrix, bias_vector):
    """
    Forward pass for an affine layer.
    
    Inputs:
    - input_data: numpy array of shape (N, d_1, ..., d_k)
    - weight_matrix: numpy array of shape (D, M)
    - bias_vector: numpy array of shape (M,)
    
    Returns:
    - output: numpy array of shape (N, M)
    - cache: Tuple of (input_data, weight_matrix, bias_vector)
    """
    reshaped_input = input_data.reshape(input_data.shape[0], -1)
    output = np.dot(reshaped_input, weight_matrix) + bias_vector
    cache = (input_data, weight_matrix, bias_vector)
    return output, cache

def affine_transform_backward(dout, cache):
    """
    Backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, shape (N, M)
    - cache: Tuple of (input_data, weight_matrix, bias_vector)

    Returns:
    - dinput: Gradient with respect to input, shape (N, d_1, ..., d_k)
    - dweight: Gradient with respect to weights, shape (D, M)
    - dbias: Gradient with respect to biases, shape (M,)
    """
    input_data, weight_matrix, bias_vector = cache
    reshaped_input = input_data.reshape(input_data.shape[0], -1)
    dbias = np.sum(dout, axis=0)
    dweight = np.dot(reshaped_input.T, dout)
    dreshaped_input = np.dot(dout, weight_matrix.T)
    dinput = dreshaped_input.reshape(input_data.shape)
    return dinput, dweight, dbias

def relu_activation_forward(input_data):
    """
    Forward pass for ReLU activation.

    Input:
    - input_data: numpy array of any shape

    Returns:
    - output: numpy array of same shape as input_data
    - cache: input_data
    """
    output = np.maximum(0, input_data)
    cache = input_data
    return output, cache

def relu_activation_backward(dout, cache):
    """
    Backward pass for ReLU activation.

    Input:
    - dout: Upstream derivatives, same shape as cache
    - cache: Input data, same shape as dout

    Returns:
    - dinput: Gradient with respect to input
    """
    input_data = cache
    dinput = dout * (input_data > 0)
    return dinput

def batch_normalization_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    
    Inputs:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with mode, eps, momentum, running_mean, running_var

    Returns:
    - out: Output, shape (N, D)
    - cache: Values needed for backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    if mode == 'train':
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        x_normalized = (x - mean) / np.sqrt(var + eps)
        out = gamma * x_normalized + beta

        cache = (x, x_normalized, mean, var, gamma, beta, eps)
    elif mode == 'test':
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        cache = None
    else:
        raise ValueError('Invalid forward batch normalization mode "%s"' % mode)

    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def batch_normalization_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, shape (N, D)
    - cache: Variables from batch_normalization_forward

    Returns:
    - dx: Gradient with respect to inputs, shape (N, D)
    - dgamma: Gradient with respect to scale parameter, shape (D,)
    - dbeta: Gradient with respect to shift parameter, shape (D,)
    """
    x, x_normalized, mean, var, gamma, beta, eps = cache

    N, D = dout.shape
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_normalized, axis=0)

    dxhat = dout * gamma
    dvar = np.sum(dxhat * (x - mean) * -0.5 * np.power(var + eps, -1.5), axis=0)
    dmean = np.sum(dxhat * -1 / np.sqrt(var + eps), axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)

    dx = dxhat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / N + dmean / N

    return dx, dgamma, dbeta

def dropout_layer_forward(x, dropout_param):
    """
    Forward pass for dropout.

    Inputs:
    - x: Input data, any shape
    - dropout_param: Dictionary with p, mode, seed

    Returns:
    - out: Array same shape as x
    - cache: Tuple of (dropout_param, mask)
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    if mode == 'train':
        mask = (np.random.rand(*x.shape) < p) / p
        out = x * mask
    elif mode == 'test':
        out = x
        mask = None
    else:
        raise ValueError('Invalid dropout mode "%s"' % mode)

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)
    return out, cache

def dropout_layer_backward(dout, cache):
    """
    Backward pass for dropout.

    Inputs:
    - dout: Upstream derivatives, any shape
    - cache: (dropout_param, mask) from dropout_layer_forward

    Returns:
    - dx: Gradient with respect to input
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    if mode == 'train':
        dx = dout * mask
    elif mode == 'test':
        dx = dout
    else:
        raise ValueError('Invalid dropout mode "%s"' % mode)

    return dx

def svm_loss_function(scores, labels):
    """
    Computes the SVM loss and gradient.

    Inputs:
    - scores: Input data, shape (N, C)
    - labels: Vector of labels, shape (N,)

    Returns:
    - loss: Scalar value
    - dscores: Gradient of the loss with respect to scores
    """
    N = scores.shape[0]
    correct_scores = scores[np.arange(N), labels]
    margins = np.maximum(0, scores - correct_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), labels] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dscores = np.zeros_like(scores)
    dscores[margins > 0] = 1
    dscores[np.arange(N), labels] -= num_pos
    dscores /= N
    return loss, dscores

def softmax_loss_function(scores, labels):
    """
    Computes the softmax loss and gradient.

    Inputs:
    - scores: Input data, shape (N, C)
    - labels: Vector of labels, shape (N,)

    Returns:
    - loss: Scalar value
    - dscores: Gradient of the loss with respect to scores
    """
    shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    N = scores.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), labels])) / N
    dscores = probs.copy()
    dscores[np.arange(N), labels] -= 1
    dscores /= N
    return loss, dscores

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """
    Forward pass for affine layer, batch normalization, and ReLU activation.

    Inputs:
    - x: Input data, shape (N, D1)
    - w: Weights, shape (D1, D2)
    - b: Biases, shape (D2)
    - gamma: Scale parameter, shape (D2)
    - beta: Shift parameter, shape (D2)
    - bn_param: Batch normalization parameters

    Returns:
    - out: Output from ReLU, shape (N, D2)
    - cache: Values needed for backward pass
    """
    affine_out, fc_cache = affine_transform_forward(x, w, b)
    bn_out, bn_cache = batch_normalization_forward(affine_out, gamma, beta, bn_param)
    relu_out, relu_cache = relu_activation_forward(bn_out)
    cache = (fc_cache, bn_cache, relu_cache)
    return relu_out, cache

def affine_bn_relu_backward(dout, cache):
    """
    Backward pass for affine layer, batch normalization, and ReLU activation.

    Inputs:
    - dout: Upstream derivatives, shape (N, D2)
    - cache: Values from forward pass

    Returns:
    - dx: Gradient with respect to input, shape (N, D1)
    - dw: Gradient with respect to weights, shape (D1, D2)
    - db: Gradient with respect to biases, shape (D2)
    - dgamma: Gradient with respect to scale parameter, shape (D2)
    - dbeta: Gradient with respect to shift parameter, shape (D2)
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn_out = relu_activation_backward(dout, relu_cache)
    daffine_out, dgamma, dbeta = batch_normalization_backward(dbn_out, bn_cache)
    dx, dw, db = affine_transform_backward(daffine_out, fc_cache)
    return dx, dw, db, dgamma, dbeta
