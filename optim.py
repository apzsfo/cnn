import numpy as np

"""
This file implements various first-order update rules that are commonly used for
training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(weights, gradient, config=None):

Inputs:
  - weights: A numpy array giving the current weights.
  - gradient: A numpy array of the same shape as weights giving the gradient of the
    loss with respect to the weights.
  - config: A dictionary containing hyperparameter values such as learning rate,
    momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_weights: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not perform
well; however the default values of the other hyperparameters should work well
for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating weights and
setting next_weights equal to weights.
"""


def sgd(weights, gradient, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    weights -= config['learning_rate'] * gradient
    return weights, config


def sgd_momentum(weights, gradient, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as weights and gradient used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    velocity = config.get('velocity', np.zeros_like(weights))
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the momentum update formula. Return the updated weights
    #   as next_weights, and the updated velocity as velocity.
    # ================================================================ #
    alpha = config['momentum']
    epsilon = config['learning_rate']
    velocity = alpha * velocity - epsilon * gradient
    next_weights = weights + velocity
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    config['velocity'] = velocity

    return next_weights, config

def sgd_nesterov_momentum(weights, gradient, config=None):
    """
    Performs stochastic gradient descent with Nesterov momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as weights and gradient used to store a moving
      average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    velocity = config.get('velocity', np.zeros_like(weights))
    
    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement the momentum update formula. Return the updated weights
    #   as next_weights, and the updated velocity as velocity.
    # ================================================================ #
    alpha = config['momentum']
    epsilon = config['learning_rate']
    old_velocity = velocity
    velocity = alpha * velocity - epsilon * gradient
    next_weights = weights + velocity + alpha * (velocity - old_velocity)
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    config['velocity'] = velocity

    return next_weights, config

def rmsprop(weights, gradient, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(weights))

    next_weights = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement RMSProp. Store the next value of weights as next_weights. 
    #   You need to also store in config['cache'] the moving average of the second
    #   moment gradients, so they can be used for future gradients. Concretely,
    #   config['cache'] corresponds to "cache" in the lecture notes.
    # ================================================================ #
    cache = config['cache']
    learning_rate = config['learning_rate']
    epsilon = config['epsilon']
    decay_rate = config['decay_rate']
    cache = decay_rate * cache + (1 - decay_rate) * gradient * gradient
    next_weights = weights - learning_rate * gradient / (np.sqrt(cache + epsilon))
    config['cache'] = cache
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return next_weights, config


def adam(weights, gradient, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(weights))
    config.setdefault('v', np.zeros_like(weights))
    config.setdefault('t', 0)
    
    next_weights = None

    # ================================================================ #
    # YOUR CODE HERE:
    #   Implement Adam. Store the next value of weights as next_weights. 
    #   You need to also store in config['m'] the moving average of the first
    #   moment gradients, and in config['v'] the moving average of the
    #   second moments. Finally, store in config['t'] the increasing time.
    # ================================================================ #
    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    epsilon = config['epsilon']
    m = config['m']
    v = config['v']
    t = config['t']

    t += 1
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient * gradient
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    next_weights = weights - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    config['m'] = m
    config['v'] = v
    config['t'] = t
    # ================================================================ #
    # END YOUR CODE HERE
    # ================================================================ #
    
    return next_weights, config
